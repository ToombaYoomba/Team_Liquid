from pathlib import Path
import polars as pl
from collections import defaultdict

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/metrics")
OUTPUT_DIR.mkdir(exist_ok=True)

HITS_FILES = {
    "v1": "2022_yandex_metrika_hits.parquet",
    "v2": "2024_yandex_metrika_hits.parquet",
}
VISITS_FILES = {
    "v1": "2022_yandex_metrika_visits.parquet",
    "v2": "2024_yandex_metrika_visits.parquet",
}

CHUNK_SIZE = 100_000

def visits_with_n_hits(df: pl.DataFrame) -> pl.DataFrame:
    """Добавляет колонку hits_count = длина watchIDs."""
    if df.is_empty():
        return df
    return (
        df.with_columns(
            pl.col("ym:s:watchIDs")
            .str.json_decode(dtype=pl.List(pl.Utf8))
            .list.eval(pl.element().cast(pl.UInt64, strict=False))
            .alias("watchids_parsed")
        )
        .with_columns(
            pl.col("watchids_parsed").list.len().alias("hits_count")
        )
    )

def add_goals(df: pl.DataFrame) -> pl.DataFrame:
    """
    Распарсить ym:s:goalsID (JSON-список) → list[UInt64] в колонке goals_ids.
    Пустые/отсутствующие цели превращаются в пустой список.
    """
    if df.is_empty():
        return df

    return (
        df.with_columns(
            pl.col("ym:s:goalsID")
            .str.json_decode(dtype=pl.List(pl.Utf8))
            .list.eval(pl.element().cast(pl.UInt64, strict=False))
            .alias("goals_ids")
        )
    )





def compute_advanced_metrics(version: str) -> dict:
    """Полный набор метрик для одной версии"""
    print(f"\n=== ГЛУБОКИЙ АНАЛИЗ {version} ===")
    
    visits_file = DATA_DIR / VISITS_FILES[version]
    hits_file = DATA_DIR / HITS_FILES[version]
    
    metrics = defaultdict(float)
    

    # БАЗОВЫЕ МЕТРИКИ (VISITS)

    print("Базовые метрики...")
    visits_lf = (
        pl.scan_parquet(visits_file)
        .select([
            "ym:s:visitID", "ym:s:counterID", "ym:s:watchIDs", "ym:s:isNewUser", 
            "ym:s:date", "ym:s:dateTime", "ym:s:dateTimeUTC", "ym:s:visitDuration", "ym:s:startURL",
            "ym:s:endURL", "ym:s:goalsID"
        ])
    )
    
    total_visits = 0
    sum_visit_duration = 0.0
    new_users = 0
    bounce_visits = 0
    avg_duration = 0.0
    visit_durations = []

    batches = visits_lf.collect(streaming=True).iter_slices(CHUNK_SIZE)
    for batch in batches:
        visits = batch
        chunk_size = visits.height
        total_visits += chunk_size

        new_users += visits.select(pl.col("ym:s:isNewUser").sum()).item()

        visits_hc = visits_with_n_hits(visits)

        chunk_bounce = (
            visits_hc
            .select(pl.col("hits_count").eq(1).sum())
            .item()
        )

        sum_visit_duration += (
            visits
            .select(pl.col("ym:s:visitDuration").cast(pl.Float64, strict=False).sum())
            .item()
        )

        bounce_visits += chunk_bounce
        
        print(f"  Visits: {total_visits:,}")

    avg_session_duration = (
        float(sum_visit_duration / total_visits) if total_visits else 0.0
    )
    
    # HITS
    hits_lf = (
        pl.scan_parquet(hits_file)
        .select([
            "ym:pv:watchID", "ym:pv:pageViewID", "ym:pv:URL", 
            "ym:pv:dateTime", "ym:pv:clientID"
        ])
    )
    
    total_hits = 0
    unique_users = 0
    landing_pages = defaultdict(int)
    exit_pages = defaultdict(int)
    depth_1 = depth_2plus = 0
    
    batches = hits_lf.collect(streaming=True).iter_slices(CHUNK_SIZE * 5)
    for i, batch in enumerate(batches):
        hits = batch
        chunk_hits = hits.height
        total_hits += chunk_hits

        unique_users += hits["ym:pv:clientID"].n_unique()

        top_pages = (
            hits.group_by("ym:pv:URL")
            .agg(total=pl.col("ym:pv:pageViewID").count())
            .sort("total", descending=True)
            .head(10)
        )
        for row in top_pages.iter_rows(named=True):
            landing_pages[row["ym:pv:URL"]] += row["total"]
        
        print(f"  Hits: {total_hits:,}")

    # БОЛЕЕ ГЛУБОКИЕ МЕТРИКИ
    
    # Конверсия по глубине просмотра
    depth_1 = bounce_visits
    depth_2plus = total_visits - bounce_visits
    
    # Топ лендинги и экзиты (из visits)
    top_landing_df = (
        visits
        .group_by(pl.col("ym:s:startURL").fill_null("unknown"))
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(1)
    )

    top_landing_url = top_landing_df["ym:s:startURL"][0]

    top_exit_df = (
        visits
        .group_by(pl.col("ym:s:endURL").fill_null("unknown"))
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(1)
    )
    top_exit_url = top_exit_df["ym:s:endURL"][0]

    top3 = sorted(landing_pages.items(), key=lambda kv: kv[1], reverse=True)[:3]

    top_pages_count = sum(count for _, count in top3)




    visits_goals_df = (
        pl.scan_parquet(visits_file)
        .select([
            "ym:s:visitID",
            "ym:s:dateTime",
            "ym:s:startURL",
            "ym:s:endURL",
            "ym:s:visitDuration",
            "ym:s:watchIDs",
            "ym:s:goalsID",
        ])
        .collect(streaming=True)
    )

    visits_goals_df = (
        visits_with_n_hits(visits_goals_df)
        .pipe(add_goals)
    )

    visits_with_goal = visits_goals_df.filter(
        pl.col("goals_ids").list.len() > 0
    )

    goals_expanded = (
        visits_with_goal
        .explode("goals_ids")
        .rename({"goals_ids": "goal_id"})
    )

    goal_stats = (
        goals_expanded
        .group_by("goal_id")
        .agg([
            pl.col("hits_count").mean().alias("avg_steps"),
            pl.col("ym:s:visitDuration").mean().alias("avg_duration_sec"),
        ])
    )

    overall_goal_steps = goals_expanded.select(pl.col("hits_count").mean()).item()
    overall_goal_duration = goals_expanded.select(pl.col("ym:s:visitDuration").mean()).item()

    metrics.update({
        "goal_visits_total": int(goals_expanded.height),
        "goal_avg_steps": float(overall_goal_steps),
        "goal_avg_duration_sec": float(overall_goal_duration),
    })

    goal_stats.write_parquet(OUTPUT_DIR / f"goal_stats_{version}.parquet")




    
    metrics.update({
        "total_visits": int(total_visits),
        "total_hits": int(total_hits),
        "unique_users": int(unique_users),
        "new_users": int(new_users),

        "new_user_rate": float(new_users / total_visits * 100),
        "bounce_rate": float(bounce_visits / total_visits * 100) if total_visits else 0.0,
        "pages_per_visit": float(total_hits / total_visits),
        "user_engagement": float((total_visits - bounce_visits) / total_visits * 100),

        "visits_depth_1": int(depth_1),
        "visits_depth_2plus": int(depth_2plus),
        "deep_visits_rate": float((total_visits - bounce_visits) / total_visits * 100) if total_visits else 0.0,
 
        "unique_users": int(unique_users),
        "hits_per_user": float(total_hits / unique_users) if unique_users else 0.0,
        "visits_per_user": float(total_visits / unique_users) if unique_users else 0.0,

        "avg_pages": float(total_hits / total_visits),
        "session_duration_sec": avg_session_duration,

        "top_landing_page": str(top_landing_url),
        "top_exit_page": str(top_exit_url),
        "top_pages_count": int(top_pages_count),
    })
    
    print(f"{version}: {total_visits:,} визитов | {total_hits:,} хитов | "
          f"отказы {metrics['bounce_rate']:.1f}% | глубина {metrics['avg_pages']:.1f}")
    
    return dict(metrics)







metrics_v1 = compute_advanced_metrics("v1")
metrics_v2 = compute_advanced_metrics("v2")

goal_stats_v1 = pl.read_parquet(OUTPUT_DIR / "goal_stats_v1.parquet")
goal_stats_v2 = pl.read_parquet(OUTPUT_DIR / "goal_stats_v2.parquet")

common_goals = (
        goal_stats_v1.select("goal_id")
        .join(goal_stats_v2.select("goal_id"), on="goal_id", how="inner")
        .unique()
    )

goal_stats_v1_common = goal_stats_v1.join(common_goals, on="goal_id", how="inner")
goal_stats_v2_common = goal_stats_v2.join(common_goals, on="goal_id", how="inner")

goal_stats_v1_common = goal_stats_v1_common.with_columns(pl.lit("v1").alias("version"))
goal_stats_v2_common = goal_stats_v2_common.with_columns(pl.lit("v2").alias("version"))

goal_stats_both = pl.concat([goal_stats_v1_common, goal_stats_v2_common])

goal_stats_both.write_parquet(OUTPUT_DIR / "goal_stats_common_v1_v2.parquet")



key_metrics = [
    "total_visits", "total_hits", "unique_users", "new_users",
    "new_user_rate", "bounce_rate", "pages_per_visit", 
    "deep_visits_rate", "hits_per_user", "visits_per_user"
]

comparison = pl.DataFrame({
    "metric": key_metrics,
    "v1": pl.Series("v1", [metrics_v1.get(m, 0) for m in key_metrics], dtype=pl.Float64),
    "v2": pl.Series("v2", [metrics_v2.get(m, 0) for m in key_metrics], dtype=pl.Float64)
})

comparison.write_parquet(OUTPUT_DIR / "advanced_metrics.parquet")
pl.DataFrame([{"version": "v1", **metrics_v1}, {"version": "v2", **metrics_v2}]).write_parquet(OUTPUT_DIR / "full_metrics.parquet")

