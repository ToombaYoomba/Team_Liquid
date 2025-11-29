# Team_Liquid
Это репозиторий команды TeamLiquid, реалезующей платформу для автоматизированного анализа пользовательского опыта на основе данных Яндекс.Метрики. Код нашей программы содержится в ветках репозитория: Dashboard - ветка, содержащая код визуализации проекта; LLM_testing, LLM, LLM_4files - ветки LMM-агента; make_metrics - ветка с созданием метрик на основе данных, предоставленных для анализа.
## Структура проекта
raw_data.parquet → etl (извлечение данных, ) →  embeddings → ux-анализ от ИИ → dashboard
## Инструкция к запуску дашборда
1. Скачать репозиторий git clone https://github.com/ToombaYoomba/Team_Liquid
2. Запустить файл dashboard.py из ветки Dashboard
3. Прописать в консоли pip install streamlit
4. Прописать в консоли streamlit run dashboard.py
