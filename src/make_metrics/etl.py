from pathlib import Path
from urllib.parse import urlparse
import polars as pl
from statsmodels.stats.proportion import proportions_ztest

DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HITS_FILES = {
    "v1": "2022_yandex_metrika_hits.parquet",
    "v2": "2024_yandex_metrika_hits.parquet",
}
VISITS_FILES = {
    "v1": "2022_yandex_metrika_visits.parquet",
    "v2": "2024_yandex_metrika_visits.parquet",
}

COL_HITS_WATCHID = "ym:pv:watchID"
COL_HITS_URL = "ym:pv:URL"
COL_HITS_DATETIME = "ym:pv:dateTime"

COL_VISITS_ID = "ym:s:visitID"
COL_VISITS_DATE = "ym:s:date"
COL_VISITS_WATCHIDS = "ym:s:watchIDs"

DATETIME_FMT = "%Y-%m-%d %H:%M:%S"
DATE_FMT = "%Y-%m-%d"

# Правила соответствия path -> шаг воронки
FUNNEL_RULES = {
    "landing": ["/", ],
    "list": ["/mega", "/base", "/bachelor", "/spo", "/spec", "/news", "/public", "/list" ],
    "details": ["/base/programs", "/spo/programs", "/bachelor/programs", "/spo/programs", "/programs/", ],
    "lead": ["/results/orders", "/rating"],
}

# Утилиты:

def load_hits_visits(version: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Загрузка hits/visits для одной версии, добавление колонки version."""
    hits_path = DATA_DIR / HITS_FILES[version]
    visits_path = DATA_DIR / VISITS_FILES[version]

    hits = pl.read_parquet(hits_path)
    visits = pl.read_parquet(visits_path)

    hits = hits.with_columns(pl.lit(version).alias("version"))
    visits = visits.with_columns(pl.lit(version).alias("version"))
    return hits, visits


def path_to_step(path: str | None) -> str | None:
    """Маппинг URL path -> шаг воронки по правилам FUNNEL_RULES."""
    if path is None:
        return None
    for step, patterns in FUNNEL_RULES.items():
        for p in patterns:
            if path == p or path.startswith(p):
                return step
    return None


def compute_funnel_conv(funnel_per_session: pl.DataFrame) -> pl.DataFrame:
    """Считает конверсии между шагами воронки для всех пар."""

    steps = list(FUNNEL_RULES.keys())
    pairs: list[tuple[str, str]] = list(zip(steps[:-1], steps[1:]))

    results = []

    for step_from, step_to in pairs:
        base_col = f"has_{step_from}"
        to_col = f"has_{step_to}"

        base = funnel_per_session.filter(pl.col(base_col))

        df_pair = (
            base.group_by("version")
            .agg(
                [
                    pl.count().alias("visits_from"),
                    pl.len().filter(pl.col(to_col)).alias("visits_to"),
                ]
            )
            .with_columns(
                [
                    (pl.col("visits_to") / pl.col("visits_from")).alias("cr"),
                    pl.lit(step_from).alias("step_from"),
                    pl.lit(step_to).alias("step_to"),
                ]
            )
        )

        results.append(df_pair)

    if not results:
        return pl.DataFrame([])

    funnel_conv = pl.concat(results)

    # A/B‑дельта и p‑value по конверсии для v1 vs v2
    if "v1" in funnel_conv["version"].to_list() and "v2" in funnel_conv["version"].to_list():
        rows = []
        for sf, st in zip(funnel_conv["step_from"].to_list(),
                          funnel_conv["step_to"].to_list()):
            subset = funnel_conv.filter(
                (pl.col("step_from") == sf) & (pl.col("step_to") == st)
            )

            if subset.height != 2:
                continue

            row_v1 = subset.filter(pl.col("version") == "v1").row(0, named=True)
            row_v2 = subset.filter(pl.col("version") == "v2").row(0, named=True)

            count = [row_v1["visits_to"], row_v2["visits_to"]]
            nobs = [row_v1["visits_from"], row_v2["visits_from"]]

            stat, p_value = proportions_ztest(count, nobs)

            rows.append(
                {
                    "step_from": sf,
                    "step_to": st,
                    "version_old": "v1",
                    "version_new": "v2",
                    "cr_old": row_v1["cr"],
                    "cr_new": row_v2["cr"],
                    "delta_abs": row_v2["cr"] - row_v1["cr"],
                    "delta_rel": (row_v2["cr"] - row_v1["cr"]) / row_v1["cr"]
                    if row_v1["cr"] != 0
                    else None,
                    "p_value": p_value,
                }
            )

        if rows:
            ab_df = pl.DataFrame(rows)
        else:
            ab_df = pl.DataFrame([])
    else:
        ab_df = pl.DataFrame([])

    return funnel_conv, ab_df




def main():
    # Загрузка для v1 и v2
    hits_v1, visits_v1 = load_hits_visits("v1")
    hits_v2, visits_v2 = load_hits_visits("v2")

    hits = pl.concat([hits_v1, hits_v2])
    visits = pl.concat([visits_v1, visits_v2])

    # Приведение времени и дат
    hits = hits.with_columns(
        pl.col(COL_HITS_DATETIME)
        .str.strptime(pl.Datetime, fmt=DATETIME_FMT)
        .alias("dt")
    )

    visits = visits.with_columns(
        pl.col(COL_VISITS_DATE)
        .str.strptime(pl.Date, fmt=DATE_FMT)
        .alias("date")
    )

    # Связка hits и visits по watchID
    visit_watch = visits.select(
        pl.col(COL_VISITS_ID),
        pl.col("version"),
        pl.col(COL_VISITS_WATCHIDS).explode().alias(COL_HITS_WATCHID),
    )

    hits = hits.join(
        visit_watch,
        on=[COL_HITS_WATCHID, "version"],
        how="left",
    ).rename({COL_VISITS_ID: "visit_id"})

    # Метрики по сессиям (visits + hits)
    hits_per_visit = (
        hits.group_by(["visit_id", "version"])
        .agg(
            [
                pl.count().alias("hits_count"),
                pl.col("dt").min().alias("visit_start"),
                pl.col("dt").max().alias("visit_end"),
            ]
        )
        .with_columns(
            (pl.col("visit_end") - pl.col("visit_start"))
            .dt.seconds()
            .alias("session_length_sec")
        )
    )

    sessions = visits.rename({COL_VISITS_ID: "visit_id"}).join(
        hits_per_visit, on=["visit_id", "version"], how="left"
    )

    sessions = sessions.with_columns(
        [
            pl.col("hits_count").fill_null(0),
            (pl.col("hits_count") == 1).alias("is_bounce"),
        ]
    )

    # Агрегаты по версии и дате
    version_daily = (
        sessions.group_by(["version", "date"])
        .agg(
            [
                pl.count().alias("visits"),
                pl.mean("session_length_sec").alias("avg_session_length_sec"),
                pl.mean("hits_count").alias("avg_pageviews"),
                pl.mean(pl.col("is_bounce").cast(pl.Float64)).alias("bounce_rate"),
            ]
        )
        .sort(["version", "date"])
    )

    # Метрики по страницам (hits)
    hits = hits.with_columns(
        [
            pl.col(COL_HITS_URL)
            .map_elements(lambda x: urlparse(x).path if x is not None else "")
            .alias("path")
        ]
    )

    hits = hits.sort(["visit_id", "dt"])

    hits = hits.with_columns(
        [
            pl.col("dt").shift(-1).over("visit_id").alias("next_dt"),
            pl.col("path").shift(-1).over("visit_id").alias("next_path"),
        ]
    ).with_columns(
        [
            (pl.col("next_dt") - pl.col("dt"))
            .dt.seconds()
            .fill_null(0)
            .alias("time_on_page_sec"),
            pl.col("next_path").is_null().alias("is_exit"),
        ]
    )

    page_metrics = (
        hits.group_by(["version", "path"])
        .agg(
            [
                pl.count().alias("pageviews"),
                pl.col("visit_id").n_unique().alias("unique_visits"),
                pl.mean("time_on_page_sec").alias("avg_time_on_page_sec"),
                pl.mean(pl.col("is_exit").cast(pl.Float64)).alias("exit_rate"),
            ]
        )
        .sort(["version", "pageviews"], descending=[False, True])
    )

    # Воронка по сессиям
    hits = hits.with_columns(
        pl.col("path").map_elements(path_to_step).alias("funnel_step")
    )

    funnel_per_session = (
        hits.drop_nulls("funnel_step")
        .group_by(["visit_id", "version"])
        .agg(
            [
                pl.col("funnel_step").unique().alias("steps_order"),
            ]
        )
    )

    # Флаги наличия каждого шага
    for step in FUNNEL_RULES.keys():
        funnel_per_session = funnel_per_session.with_columns(
            pl.col("steps_order").list.contains(step).alias(f"has_{step}")
        )

    # Конверсии между шагами и A/B‑дельты
    funnel_conv, funnel_ab = compute_funnel_conv(funnel_per_session)

    # Датасет сессий для ML
    sessions_for_ml = (
        hits.group_by(["visit_id", "version"])
        .agg(
            [
                pl.col("path").alias("paths_seq"),
                pl.col("funnel_step").alias("steps_seq"),
            ]
        )
        .join(
            sessions.select(
                "visit_id",
                "version",
                "session_length_sec",
                "hits_count",
                "is_bounce",
            ),
            on=["visit_id", "version"],
            how="left",
        )
        .with_columns(
            [
                pl.col("paths_seq").list.join(" > ").alias("paths_str"),
                pl.col("steps_seq").list.join(" > ").alias("steps_str"),
            ]
        )
    )

    #  Сохранение результатов
    version_daily.write_parquet(OUT_DIR / "version_daily.parquet")
    page_metrics.write_parquet(OUT_DIR / "page_metrics.parquet")
    funnel_per_session.write_parquet(OUT_DIR / "funnel_sessions.parquet")
    funnel_conv.write_parquet(OUT_DIR / "funnel_conversions.parquet")
    funnel_ab.write_parquet(OUT_DIR / "funnel_ab_test.parquet")
    sessions_for_ml.write_parquet(OUT_DIR / "sessions_for_ml.parquet")


if __name__ == "__main__":
    main()
