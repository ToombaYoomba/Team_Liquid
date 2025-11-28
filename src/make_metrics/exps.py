import pandas as pd

hits_1 = pd.read_parquet(r'C:\Users\User\Desktop\Хакатон UXанализ\Team_Liquid\data\raw\2022_yandex_metrika_hits.parquet')
print(hits_1.head(10))

visits_1 = pd.read_parquet(r'C:\Users\User\Desktop\Хакатон UXанализ\Team_Liquid\data\raw\2022_yandex_metrika_visits.parquet')
print(visits_1.head(10))

#hits_2 = pd.read_parquet(r'C:\Users\User\Desktop\Хакатон UXанализ\Team_Liquid\data\raw\2024_yandex_metrika_hits.parquet')
#print(hits_1.head(10))

#visits_2 = pd.read_parquet(r'C:\Users\User\Desktop\Хакатон UXанализ\Team_Liquid\data\raw\2024_yandex_metrika_visits.parquet')
#print(visits_1.head(10))