from pathlib import Path
import polars as pl
from collections import defaultdict
from datetime import datetime

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

def add_hits_count(df: pl.DataFrame) -> pl.DataFrame:
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

def compute_advanced_metrics(version: str) -> dict:
    """Полный набор метрик для одной версии"""
    print(f"\n=== ГЛУБОКИЙ АНАЛИЗ {version} ===")
    
    visits_file = DATA_DIR / VISITS_FILES[version]
    hits_file = DATA_DIR / HITS_FILES[version]
    
    # Аккумуляторы
    metrics = defaultdict(float)
    
    # ===============================
    # 1. БАЗОВЫЕ МЕТРИКИ (VISITS)
    # ===============================
    print("Базовые метрики...")
    visits_lf = (
        pl.scan_parquet(visits_file)
        .select([
            "ym:s:visitID", "ym:s:counterID", "ym:s:watchIDs", "ym:s:isNewUser", 
            "ym:s:date", "ym:s:dateTime", "ym:s:dateTimeUTC", "ym:s:visitDuration", "ym:s:startURL", "ym:s:endURL"
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

        # Новые пользователи
        new_users += visits.select(pl.col("ym:s:isNewUser").sum()).item()

        visits_hc = add_hits_count(visits)
        chunk_bounce = visits_hc.select(pl.col("hits_count").eq(1).sum()).item()
        bounce_visits += chunk_bounce

        # сумма длительностей визитов в секундах
        sum_visit_duration += (
            visits
            .select(pl.col("ym:s:visitDuration").cast(pl.Float64, strict=False).sum())
            .item()
        )
        
        print(f"  Visits: {total_visits:,}")

    avg_session_duration = (
        float(sum_visit_duration / total_visits) if total_visits else 0.0
    )
    
    # ===============================
    # 2. HITS & ПУТИ ПО СЕЙТУ
    # ===============================
    print(" Hits и пути...")
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

        # Уникальные пользователи в чанке (UInt64 без кастов)
        unique_users += hits["ym:pv:clientID"].n_unique()
        
        # Популярные страницы (топ-10)
        top_pages = (
            hits.group_by("ym:pv:URL")
            .agg(total=pl.col("ym:pv:pageViewID").count())
            .sort("total", descending=True)
            .head(10)
        )
        for row in top_pages.iter_rows(named=True):
            landing_pages[row["ym:pv:URL"]] += row["total"]
        
        print(f"  Hits: {total_hits:,}")

    # ===============================
    # 3. ГЛУБОКИЕ МЕТРИКИ
    # ===============================
    print("Глубокий анализ...")
    
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

    # сам URL как строка
    top_landing_url = top_landing_df["ym:s:startURL"][0]

    top_exit_df = (
        visits
        .group_by(pl.col("ym:s:endURL").fill_null("unknown"))
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(1)
    )
    top_exit_url = top_exit_df["ym:s:endURL"][0]

    # список (url, count) отсортированный по count по убыванию
    top3 = sorted(landing_pages.items(), key=lambda kv: kv[1], reverse=True)[:3]

    # сумма показов топ-3
    top_pages_count = sum(count for _, count in top3)

    
    metrics.update({
        # Базовые
        "total_visits": int(total_visits),
        "total_hits": int(total_hits),
        "unique_users": int(unique_users),
        "new_users": int(new_users),
        
        # Проценты
        "new_user_rate": float(new_users / total_visits * 100),
        "bounce_rate": float(bounce_visits / total_visits * 100) if total_visits else 0.0,
        "pages_per_visit": float(total_hits / total_visits),
        "user_engagement": float((total_visits - bounce_visits) / total_visits * 100),
        
        # Глубина
        "visits_depth_1": int(depth_1),
        "visits_depth_2plus": int(depth_2plus),
        "deep_visits_rate": float((total_visits - bounce_visits) / total_visits * 100) if total_visits else 0.0,
        
        # Трафик
        "unique_users": int(unique_users),
        "hits_per_user": float(total_hits / unique_users) if unique_users else 0.0,
        "visits_per_user": float(total_visits / unique_users) if unique_users else 0.0,
        
        # Качество трафика
        "avg_pages": float(total_hits / total_visits),
        "session_duration_sec": avg_session_duration,
        
        # Топ страницы (примерно)
        "top_landing_page": str(top_landing_url),
        "top_exit_page": str(top_exit_url),
        "top_pages_count": int(top_pages_count),  # топ-3
    })
    
    print(f"{version}: {total_visits:,} визитов | {total_hits:,} хитов | "
          f"отказы {metrics['bounce_rate']:.1f}% | глубина {metrics['avg_pages']:.1f}")
    
    return dict(metrics)


def is_priem_url(col: pl.Expr) -> pl.Expr:
    """Оставляем только URL с доменом priem.mai.ru."""
    return col.str.contains(r"^https?://priem\.mai\.ru").fill_null(False)


def compute_url_metrics(version: str, top_n: int = 200) -> pl.DataFrame:
    """
    Метрики по отдельным URL для выбранной версии:
    url, version, page_hits, page_visits, page_bounces, bounce_rate, avg_pages_per_visit.
    """
    visits_file = DATA_DIR / VISITS_FILES[version]
    hits_file = DATA_DIR / HITS_FILES[version]

    # ---------- VISITS: визиты по лендингу + bounce ----------
    visits_lf = (
        pl.scan_parquet(visits_file)
        .select([
            "ym:s:visitID",
            "ym:s:watchIDs",
            "ym:s:startURL",
        ])
    )
    visits_df = visits_lf.collect(streaming=True)
    visits_hc = add_hits_count(visits_df)

    # фильтр только URL priem.mai.ru для ПОСТРАНИЧНЫХ метрик
    visits_hc = visits_hc.filter(is_priem_url(pl.col("ym:s:startURL")))


    # группируем по стартовому URL
    visits_by_url = (
        visits_hc
        .group_by(pl.col("ym:s:startURL").fill_null("unknown").alias("url"))
        .agg([
            pl.count().alias("page_visits"),
            pl.col("hits_count").eq(1).sum().alias("page_bounces"),
        ])
    )

    # ---------- HITS: хиты по URL ----------
    hits_lf = (
        pl.scan_parquet(hits_file)
        .select(["ym:pv:URL", "ym:pv:pageViewID"])
    )
    hits_df = hits_lf.collect(streaming=True)

    # фильтр только URL priem.mai.ru для ПОСТРАНИЧНЫХ метрик
    hits_df = hits_df.filter(is_priem_url(pl.col("ym:pv:URL")))


    hits_by_url = (
        hits_df
        .group_by(pl.col("ym:pv:URL").fill_null("unknown").alias("url"))
        .agg(pl.count().alias("page_hits"))
    )

    # ---------- JOIN и расчет метрик ----------
    url_metrics = (
        hits_by_url
        .join(visits_by_url, on="url", how="outer")
        .with_columns([
            pl.col("page_hits").fill_null(0),
            pl.col("page_visits").fill_null(0),
            pl.col("page_bounces").fill_null(0),
        ])
        .with_columns([
            (pl.col("page_bounces") / pl.col("page_visits") * 100)
                .fill_null(0)
                .alias("bounce_rate"),
            (pl.col("page_hits") / pl.col("page_visits"))
                .fill_null(0)
                .alias("avg_pages_per_visit"),
            pl.lit(version).alias("version"),
        ])
        .sort("page_hits", descending=True)
        .head(top_n)
    )

    return url_metrics








# === ЗАПУСК ===
print("ГЛУБОКИЙ АНАЛИЗ МЕТРИК...")
metrics_v1 = compute_advanced_metrics("v1")
metrics_v2 = compute_advanced_metrics("v2")

print("\n📄 Считаем метрики по URL...")
url_metrics_v1 = compute_url_metrics("v1", top_n=200)
url_metrics_v2 = compute_url_metrics("v2", top_n=200)

url_metrics_v1.write_parquet(OUTPUT_DIR / "url_metrics_v1.parquet")
url_metrics_v2.write_parquet(OUTPUT_DIR / "url_metrics_v2.parquet")

print("✅ url_metrics_v1.parquet и url_metrics_v2.parquet сохранены")


# ПОЛНАЯ ТАБЛИЦА СРАВНЕНИЯ
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

print("\n=== КЛЮЧЕВЫЕ МЕТРИКИ v1 vs v2 ===")
print(comparison)

# КРАСИВАЯ ТАБЛИЦА
print("\n" + "="*80)
print("ПОЛНОЕ СРАВНЕНИЕ ВЕРСИЙ САЙТА")
print("="*80)
print("| Метрика              | v1_2022        | v2_2024        | Δ       |")
print("|----------------------|----------------|----------------|---------|")
print(f"| Визитов              | {metrics_v1['total_visits']:,}     | {metrics_v2['total_visits']:,}     | {((metrics_v2['total_visits']/metrics_v1['total_visits']-1)*100):+5.1f}% |")
print(f"| Хитов                | {metrics_v1['total_hits']:,}     | {metrics_v2['total_hits']:,}     | {((metrics_v2['total_hits']/metrics_v1['total_hits']-1)*100):+5.1f}% |")
print(f"| Уник. пользователей  | {metrics_v1['unique_users']:,}  | {metrics_v2['unique_users']:,}  | {((metrics_v2['unique_users']/metrics_v1['unique_users']-1)*100):+5.1f}% |")
print(f"| Новых пользователей  | {metrics_v1['new_user_rate']:.1f}% | {metrics_v2['new_user_rate']:.1f}% | {metrics_v2['new_user_rate']-metrics_v1['new_user_rate']:+4.1f} |")
print(f"| Отказы               | {metrics_v1['bounce_rate']:.1f}% | {metrics_v2['bounce_rate']:.1f}% | {metrics_v2['bounce_rate']-metrics_v1['bounce_rate']:+4.1f} |")
print(f"| Страниц за визит     | {metrics_v1['pages_per_visit']:.1f}  | {metrics_v2['pages_per_visit']:.1f}  | {metrics_v2['pages_per_visit']-metrics_v1['pages_per_visit']:+4.1f} |")
print(f"| Глубокие визиты      | {metrics_v1['deep_visits_rate']:.1f}% | {metrics_v2['deep_visits_rate']:.1f}% | {metrics_v2['deep_visits_rate']-metrics_v1['deep_visits_rate']:+4.1f} |")
print("="*80)

# СОХРАНЕНИЕ
comparison.write_parquet(OUTPUT_DIR / "advanced_metrics.parquet")
pl.DataFrame([{"version": "v1", **metrics_v1}, {"version": "v2", **metrics_v2}]).write_parquet(OUTPUT_DIR / "full_metrics.parquet")

print(f"\n✅ ✅ Готово! Файлы: {OUTPUT_DIR}")
print("advanced_metrics.parquet — для анализа")
print("full_metrics.parquet — полный дамп")
