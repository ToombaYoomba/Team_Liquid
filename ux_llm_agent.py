import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

from minimal_agents.agent import Agent
from minimal_agents.runner import Runner
from minimal_agents.run_config import RunConfig
from util.adk_custom_model_provider import CustomModelProvider

from mcp_ux_server import _load_full_metrics_analysis, _load_parquet_summary_direct

load_dotenv()
folder_id = os.environ["folder_id"]
api_key = os.environ["api_key"]

model = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"

client = AsyncOpenAI(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=api_key,
    project=folder_id
)

PROMPT_TEMPLATE = """
Ты — опытный UX аналитик. Ты получаешь данные сравнения двух версий продукта (V1 и V2).

ДАННЫЕ ДЛЯ АНАЛИЗА:
{metrics_data}

ТВОЯ ЗАДАЧА:
1. Выделить метрики, которые изменились более чем на 20% между версиями
2. Проанализировать проблему, связанную с изменением каждой метрики
3. Предложить конкретное решение для UI/UX дизайнера

ПРАВИЛА АНАЛИЗА:
- Анализируй ТОЛЬКО метрики с изменением >20% (абсолютное значение)
- Для каждой метрики определи единицы измерения (секунды, проценты, количество и т.д.)
- Сформулируйте понятную проблему на основе изменения метрики
- Предложите КОНКРЕТНОЕ решение для улучшения UX

ФОРМАТ ОТВЕТА ОБЯЗАТЕЛЕН:
{{
  "analysis": [
    {{
      "metric": "название_метрики",
      "unit": "единица_измерения",
      "version_a": число,
      "version_b": число, 
      "relative_change": число,
      "insight": "подробное описание проблемы",
      "solution": "конкретное предложение для дизайнера"
    }}
  ]
}}

ПРИМЕРЫ ЕДИНИЦ ИЗМЕРЕНИЯ:
- секунды
- проценты
- просмотры
- сессии
- минуты
- клики
- пользователи

ПРИМЕРЫ ФОРМАТА:
{{
  "metric": "time_on_page",
  "unit": "секунды",
  "version_a": 15.2,
  "version_b": 23.7, 
  "relative_change": 55.9,
  "insight": "Пользователи проводят больше времени на странице...",
  "solution": "Упростить навигацию и добавить четкие призывы к действию"
}}

{{
  "metric": "conversion_rate", 
  "unit": "проценты",
  "version_a": 4.2,
  "version_b": 3.1,
  "relative_change": -26.2,
  "insight": "Конверсия снизилась...",
  "solution": "Увеличить контрастность кнопки призыва к действию"
}}

ВАЖНО: 
- version_a и version_b должны быть ЧИСЛАМИ (без единиц измерения)
- Единицы измерения указывай ТОЛЬКО в поле "unit"
- Будь конкретен в решениях! Не пиши общие фразы.
"""

agent = Agent(
    name="UX_Metrics_Analyzer",
    instructions="""Ты — эксперт по UX аналитике. Строго следуй правилам:
1. Анализируй только метрики с изменением >20%
2. Определяй единицы измерения для каждой метрики в поле "unit"
3. version_a и version_b должны быть только числами
4. Формулируй конкретные проблемы на основе данных
5. Предлагай практические решения для дизайнеров
6. Возвращай только JSON в указанном формате""",
    model=model
)

rc = RunConfig(model_provider=CustomModelProvider(model, client))

async def run_metrics_analysis(metrics_file: str):
    print(f"Loading data from {metrics_file}...")
    
    json_data = _load_full_metrics_analysis(metrics_file)
    
    data = json.loads(json_data)
    
    prompt = PROMPT_TEMPLATE.format(
        metrics_data=json_data
    )

    print("Running LLM analysis...")
    result = await Runner.run(agent, input=prompt, run_config=rc)
    return result.final_output

async def run_analysis_simple(file_a: str, file_b: str):
    json_a = _load_parquet_summary_direct(file_a)
    json_b = _load_parquet_summary_direct(file_b)

    prompt = PROMPT_TEMPLATE.format(
        version_a=json_a,
        version_b=json_b
    )

    result = await Runner.run(agent, input=prompt, run_config=rc)
    return result.final_output

if __name__ == "__main__":
    metrics_file = "data/full_metrics.parquet"
    
    print("UX metrics analysis started...")
    
    out = asyncio.run(run_metrics_analysis(metrics_file))

    try:
        parsed = json.loads(out)
        with open("ux_metrics_analysis.json", "w", encoding="utf8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        
        analysis = parsed.get("analysis", [])
        total_problems = len(analysis)
        
        print("Report saved to ux_metrics_analysis.json")
                    
    except json.JSONDecodeError:
        print("Error: Invalid response from LLM")