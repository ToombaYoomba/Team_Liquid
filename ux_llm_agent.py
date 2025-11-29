import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

from minimal_agents.agent import Agent
from minimal_agents.runner import Runner
from minimal_agents.run_config import RunConfig
from util.adk_custom_model_provider import CustomModelProvider

from mcp_ux_server import _load_combined_ux_data, _load_parquet_summary_direct

load_dotenv()
folder_id = os.environ["folder_id"]
api_key = os.environ["api_key"]

model = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"

client = AsyncOpenAI(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=api_key,
    project=folder_id
)

REL_CHANGE_THRESHOLD = 40  # ОБЯЗАТЕЛЬНО БЕЗ ДЕСЯТИЧНОЙ ТОЧКИ ИНАЧЕ МЯСО
Z_SCORE_THRESHOLD = 2.0

PROMPT_TEMPLATE = """
Ты получаешь два набора данных (VERSION_A и VERSION_B) о поведении пользователей на сайте. 

ТВОЯ ЗАДАЧА: Найти ВСЕ метрики, которые изменились более чем на 40% между версиями.

ОСОБОЕ ВНИМАНИЕ УДЕЛИ:
- Изменения в популярности конкретных URL и категорий страниц
- Изменения в глубине сессий и bounce rate
- Изменения в распределении по типам контента

ДАННЫЕ VERSION_A:
{version_a}

ДАННЫЕ VERSION_B:
{version_b}

АНАЛИЗИРУЙ СЛЕДУЮЩЕЕ:
- Количество просмотров по КОНКРЕТНЫМ URL (из top_urls)
- Распределение по КАТЕГОРИЯМ URL (url_category_analysis)  
- Статистики сессий (глубина, bounce rate)
- Общие метрики (total_hits, total_visits)

ДЛЯ КАЖДОЙ МЕТРИКИ РАССЧИТАЙ:
1. Абсолютное изменение (версия_B - версия_A)
2. Относительное изменение в % ((версия_B - версия_A) / версия_A * 100)

ЕСЛИ НАЙДЕШЬ ИЗМЕНЕНИЯ >40% - включи их в отчет.
ЕСЛИ НЕТ - верни пустой список.

Формат ОБЯЗАТЕЛЕН:
{{
  "анализ_результатов": {{
    "общее_сообщение": "строка",
    "всего_значимых_изменений": число,
    "статус": "есть_значимые_изменения/нет_значимых_изменений"
  }},
  "значимые_изменения": [
    {{
      "метрика": "строка_с_понятным_названием", 
      "версия_A": число,
      "версия_B": число, 
      "абсолютное_изменение": число,
      "относительное_изменение_%": число,
      "значимость": "высокая/средняя/низкая",
      "интерпретация": "краткое_объяснение_что_это_может_означать"
    }}
  ]
}}
"""

agent = Agent(
    name="UX_MCP_Analyzer",
    instructions="Ты — аналитик UX, выполняй анализ строго по инструкции и возвращай только JSON.",
    model=model
)

rc = RunConfig(model_provider=CustomModelProvider(model, client))

async def run_analysis(
    hits_file_a: str, 
    visits_file_a: str, 
    hits_file_b: str, 
    visits_file_b: str
):
    """
    Основная функция анализа с двумя файлами на версию:
    - Загружает комбинированные данные hits + visits для обеих версий
    - Формирует промпт для LLM
    - Запускает агента
    """
    print(f"Загрузка данных...")
    
    json_a = _load_combined_ux_data(hits_file_a, visits_file_a)
    json_b = _load_combined_ux_data(hits_file_b, visits_file_b)

    with open("debug_version_a.json", "w", encoding="utf8") as f:
        f.write(json_a)
    with open("debug_version_b.json", "w", encoding="utf8") as f:
        f.write(json_b)

    data_a = json.loads(json_a)
    data_b = json.loads(json_b)
    
    print("Проверка URL анализа:")
    if 'url_category_analysis' in data_a['hits_analysis']:
        print("Категории URL Version A:", data_a['hits_analysis']['url_category_analysis'])
        print("Категории URL Version B:", data_b['hits_analysis']['url_category_analysis'])
    
    if 'top_urls' in data_a['hits_analysis']:
        print("Топ URL Version A (первые 3):", data_a['hits_analysis']['top_urls'][:3])
        print("Топ URL Version B (первые 3):", data_b['hits_analysis']['top_urls'][:3])

    prompt = PROMPT_TEMPLATE.format(
        version_a=json_a,
        version_b=json_b,
        rel_thr=REL_CHANGE_THRESHOLD,
        z_thr=Z_SCORE_THRESHOLD
    )

    print("Запуск LLM анализа...")
    result = await Runner.run(agent, input=prompt, run_config=rc)
    return result.final_output

async def run_analysis_simple(file_a: str, file_b: str):
    """
    Старая функция для обратной совместимости (один файл на версию)
    """
    json_a = _load_parquet_summary_direct(file_a)
    json_b = _load_parquet_summary_direct(file_b)

    prompt = PROMPT_TEMPLATE.format(
        version_a=json_a,
        version_b=json_b,
        rel_thr=REL_CHANGE_THRESHOLD,
        z_thr=Z_SCORE_THRESHOLD
    )

    result = await Runner.run(agent, input=prompt, run_config=rc)
    return result.final_output

if __name__ == "__main__":
    hits_a = "data/v1_hits.parquet"
    visits_a = "data/v1_visits.parquet"
    hits_b = "data/v2_hits.parquet"
    visits_b = "data/v2_visits.parquet"
    
    print("Анализ UX данных...")
    
    out = asyncio.run(run_analysis(hits_a, visits_a, hits_b, visits_b))

    try:
        parsed = json.loads(out)
        with open("ux_report_llm_significant.json", "w", encoding="utf8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        
        # Анализ результата
        changes = parsed.get("значимые_изменения", [])
        if changes:
            print(f"Найдено {len(changes)} значимых изменений (>40%)")
            for change in changes[:3]:  # покажем первые 3
                metric = change.get('метрика', 'N/A')
                change_pct = change.get('относительное_изменение_%', 0)
                print(f"   - {metric}: {change_pct:+.1f}%")
        else:
            print("   Значимых изменений не обнаружено (>40%)")
            print("   Проверьте отладочные файлы debug_version_*.json")
        
        print("  Отчет сохранен в ux_report_llm_significant.json")
                    
    except json.JSONDecodeError:
        print("  Ошибка: некорректный ответ от LLM")
        print("Ответ:", out)