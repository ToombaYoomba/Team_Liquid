import polars as pl
from pathlib import Path

DATA_DIR = Path("data/raw")

def load_hits_visits(version: str):
    hits = pl.read_parquet(DATA_DIR / f"{version}_yandex_metrika_hits.parquet")
    visits = pl.read_parquet(DATA_DIR / f"{version}_yandex_metrika_hits.parquet")
    hits = hits.with_columns(pl.lit(version).alias("version"))
    visits = visits.with_columns(pl.lit(version).alias("version"))
    return hits, visits

hits_v1, visits_v1 = load_hits_visits("2022")
hits_v2, visits_v2 = load_hits_visits("2024")

hits = pl.concat([hits_v1, hits_v2])
visits = pl.concat([visits_v1, visits_v2])




hits = hits.with_columns([
    pl.col("ym:pv:dateTime").str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S").alias("dt")  # имя и формат подставьте свои
])

visits = visits.with_columns([
    pl.col("ym:s:date").str.strptime(pl.Date, fmt="%Y-%m-%d").alias("date")
])




# Разворачиваем массив watchIDs в visits
visit_watch = visits.select(
    pl.col("ym:s:visitID"),
    pl.col("version"),
    pl.col("ym:s:watchIDs").explode().alias("ym:pv:watchID")
)

# Джойним к hits по watchID + counterID (на всякий случай)
hits = hits.join(
    visit_watch,
    on=["ym:pv:watchID", "version"],  # при необходимости добавьте counterID
    how="left"
)




# Считаем кол-во хитов по visitID
hits_per_visit = hits.group_by(["ym:s:visitID", "version"]).agg([
    pl.count().alias("hits_count"),
    pl.col("dt").min().alias("visit_start"),
    pl.col("dt").max().alias("visit_end")
])

sessions = visits.join(
    hits_per_visit,
    on=["ym:s:visitID", "version"],
    how="left"
).with_columns([
    (pl.col("visit_end") - pl.col("visit_start")).dt.seconds().alias("session_length_sec"),
    (pl.col("hits_count") == 1).alias("is_bounce")
])






version_daily = sessions.group_by(["version", "date"]).agg([
    pl.count().alias("visits"),
    pl.col("ym:s:visitID").n_unique().alias("unique_visits"),
    pl.mean("session_length_sec").alias("avg_session_length"),
    pl.mean("hits_count").alias("avg_pageviews"),
    pl.mean(pl.col("is_bounce").cast(pl.Int8)).alias("bounce_rate")
])





from urllib.parse import urlparse

hits = hits.with_columns([
    pl.col("ym:pv:URL").map_elements(
        lambda x: urlparse(x).path if x is not None else ""
    ).alias("path")
])



hits = hits.sort(["ym:s:visitID", "dt"])

hits = hits.with_columns([
    pl.col("dt").shift(-1).over("ym:s:visitID").alias("next_dt"),
    pl.col("path").shift(-1).over("ym:s:visitID").alias("next_path")
]).with_columns([
    (pl.col("next_dt") - pl.col("dt")).dt.seconds().fill_null(0).alias("time_on_page_sec"),
    # хит последний в сессии, если нет next_path
    pl.col("next_path").is_null().alias("is_exit")
])





page_metrics = hits.group_by(["version", "path"]).agg([
    pl.count().alias("pageviews"),
    pl.col("ym:s:visitID").n_unique().alias("unique_visits"),
    pl.mean("time_on_page_sec").alias("avg_time_on_page"),
    pl.mean(pl.col("is_exit").cast(pl.Int8)).alias("exit_rate")
])




FUNNEL_RULES = {
    "landing": ["/", "/index"],
    "list": ["/programs", "/search"],
    "details": ["/programs/", "/specialty/"],  # по startswith
    "lead": ["/order", "/form"]
}

def path_to_step(path: str) -> str | None:
    if path is None:
        return None
    for step, patterns in FUNNEL_RULES.items():
        for p in patterns:
            if path == p or path.startswith(p):
                return step
    return None

hits = hits.with_columns([
    pl.col("path").map_elements(path_to_step).alias("funnel_step")
])





funnel_per_session = (hits
    .drop_nulls("funnel_step")
    .group_by(["ym:s:visitID", "version"])
    .agg([
        pl.col("funnel_step").unique().alias("steps_order")  # грубо, можно лучше
    ])
)

# Флаги достижения шагов
for step in FUNNEL_RULES.keys():
    funnel_per_session = funnel_per_session.with_columns(
        pl.col("steps_order").list.contains(step).alias(f"has_{step}")
    )



import itertools

def conv(step_from, step_to):
    base = funnel_per_session.filter(pl.col(f"has_{step_from}"))
    success = base.filter(pl.col(f"has_{step_to}"))
    return (base
        .group_by("version")
        .agg([
            pl.count().alias(f"visits_{step_from}"),
            pl.len().filter(pl.col(f"has_{step_to}")).alias(f"visits_{step_to}"),
            (pl.len().filter(pl.col(f"has_{step_to}")) / pl.count()).alias("cr")
        ])
    )

conv_landing_list = conv("landing", "list")
pairs = [
    ("landing", "list"),
    ("list", "details"),
    ("details", "lead"),
]

results = []
for s_from, s_to in pairs:
    df_pair = conv(s_from, s_to)  # твоя функция conv
    df_pair = df_pair.with_columns([
        pl.lit(s_from).alias("step_from"),
        pl.lit(s_to).alias("step_to")
    ])
    results.append(df_pair)

funnel_conv = pl.concat(results)



#А/В сравнение для bounce_rate
from statsmodels.stats.proportion import proportions_ztest

# Берём агрегаты
br = version_daily.group_by("version").agg([
    (pl.col("is_bounce").sum()).alias("bounces"),
    pl.count().alias("visits")
])

b1 = int(br.filter(pl.col("version")=="v1")["bounces"][0])
n1 = int(br.filter(pl.col("version")=="v1")["visits"][0])
b2 = int(br.filter(pl.col("version")=="v2")["bounces"][0])
n2 = int(br.filter(pl.col("version")=="v2")["visits"][0])

stat, p_value = proportions_ztest([b1, b2], [n1, n2])






#Подготовка для ML
sessions_for_ml = (hits
    .group_by(["ym:s:visitID", "version"])
    .agg([
        pl.col("path").alias("paths_seq"),
        pl.col("funnel_step").alias("steps_seq"),
    ])
    .join(sessions.select(
        "ym:s:visitID", "version", "session_length_sec", "hits_count", "is_bounce"
    ), on=["ym:s:visitID", "version"], how="left")
    .with_columns([
        pl.col("paths_seq").list.join(" > ").alias("paths_str"),
        pl.col("steps_seq").list.join(" > ").alias("steps_str")
    ])
)




#Сохраняем результаты
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

version_daily.write_parquet(OUT / "version_daily.parquet")
page_metrics.write_parquet(OUT / "page_metrics.parquet")
funnel_per_session.write_parquet(OUT / "funnel_sessions.parquet")
sessions_for_ml.write_parquet(OUT / "sessions_for_ml.parquet")
