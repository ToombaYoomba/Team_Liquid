from urllib.parse import urlparse
import polars as pl

HITS_PATH = "data/raw/2024_yandex_metrika_hits.parquet"

FUNNEL_RULES = {
    "landing": ["/", ],
    "list": ["/mega", "/base", "/bachelor", "/spo", "/spec", "/news", "/public" ],
    "details": ["/base/programs", "/spo/programs", "/bachelor/programs", "/spo/programs", "/programs/", ],
    "lead": ["/list", "/results/orders", "/rating"],
}

def path_to_step(path: str | None) -> str | None:
    if path is None:
        return None
    for step, patterns in FUNNEL_RULES.items():
        for p in patterns:
            if path == p or path.startswith(p):
                return step
    return None

def main():
    # 1. Берём выборку URL из hits [web:45]
    df = pl.read_parquet(HITS_PATH)

    df = df.select("ym:pv:URL").drop_nulls()  # только URL

    # Выбираем уникальные топ‑N по встречаемости
    sample = (
        df.group_by("ym:pv:URL")
        .agg(pl.count().alias("cnt"))
        .sort("cnt", descending=True)
        .head(15)
        .with_columns(
            pl.col("ym:pv:URL")
            .map_elements(lambda u: urlparse(u).path if u is not None else "")
            .alias("path")
        )
    )

    # Применяем path_to_step и смотрим результат
    sample = sample.with_columns(
        pl.col("path").map_elements(path_to_step).alias("funnel_step")
    )

    # Печатаем/сохраняем для ручной проверки
    print(sample.sort("funnel_step", nulls_last=True))

    # Если удобнее — сохранить в CSV и открыть в Excel
    sample.write_csv("debug_funnel_sample.csv")

if __name__ == "__main__":
    main()
