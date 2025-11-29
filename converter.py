import pandas as pd

# Список файлов для конвертации
files = ['v1_hits.parquet', 'v1_visits.parquet', 'v2_hits.parquet', 'v2_visits.parquet']

for parquet_file in files:
    df = pd.read_parquet(parquet_file)
    csv_name = parquet_file.replace('.parquet', '_converted.csv')
    df.to_csv(csv_name, index=False, encoding='utf-8')
    print(f"{parquet_file} -> {csv_name}")