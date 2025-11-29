import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

from minimal_agents.agent import Agent
from minimal_agents.runner import Runner
from minimal_agents.run_config import RunConfig
from util.adk_custom_model_provider import CustomModelProvider

from mcp_ux_server import _load_full_metrics_analysis, _load_parquet_summary_direct, _load_goals_analysis, _read_excel_file

load_dotenv()
folder_id = os.environ["folder_id"]
api_key = os.environ["api_key"]

model = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"

client = AsyncOpenAI(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=api_key,
    project=folder_id
)

PROMPT_TEMPLATE_METRICS = """
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

ВАЖНО: 
- version_a и version_b должны быть ЧИСЛАМИ (без единиц измерения)
- Единицы измерения указывай ТОЛЬКО в поле "unit"
- Будь конкретен в решениях! Не пиши общие фразы.
"""

PROMPT_TEMPLATE_GOALS = """
Ты — опытный UX аналитик. Ты получаешь данные по достижению целей на сайте для двух версий (V1 и V2).

ДАННЫЕ ПО ЦЕЛЯМ:
{goals_data}

ОПИСАНИЯ ЦЕЛЕЙ:
{goals_descriptions}

ТВОЯ ЗАДАЧА:
1. Сопоставить goal_id из данных с номерами целей из описаний
2. Проанализировать изменения во времени достижения (avg_duration_sec) и количестве шагов (avg_steps) для каждой цели
3. Выделить цели, где произошли значительные изменения (>20%)
4. Проанализировать проблемы в пользовательском пути
5. Предложить конкретные решения для улучшения конверсии

ФОРМАТ ОТВЕТА ОБЯЗАТЕЛЕН:
{{
  "analysis": [
    {{
      "goal_id": "номер_цели",
      "goal_name": "название_цели_из_excel",
      "goal_description": "описание_цели_из_excel",
      "metrics": [
        {{
          "metric": "avg_steps",
          "unit": "шаги",
          "version_a": число,
          "version_b": число,
          "relative_change": число,
          "insight": "анализ изменения количества шагов",
          "solution": "решение для упрощения пути"
        }},
        {{
          "metric": "avg_duration_sec", 
          "unit": "секунды",
          "version_a": число,
          "version_b": число,
          "relative_change": число,
          "insight": "анализ изменения времени достижения",
          "solution": "решение для оптимизации времени"
        }}
      ]
    }}
  ]
}}

ВАЖНО:
- Сопоставляй goal_id из данных с колонкой "Номер цели" из Excel
- Используй "Название цели" и "Описание" из Excel для названия и описания цели
- Анализируй обе метрики (avg_steps и avg_duration_sec) для каждой цели
- Фокусируйся на целях с изменениями >20%
- Предлагай конкретные UX решения на основе типа цели
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
    
    prompt = PROMPT_TEMPLATE_METRICS.format(
        metrics_data=json_data
    )

    print("Running LLM analysis...")
    result = await Runner.run(agent, input=prompt, run_config=rc)
    return result.final_output

async def run_goals_analysis(goals_file: str, goals_descriptions_file: str):
    print(f"Loading goals data from {goals_file}...")
    goals_json_data = _load_goals_analysis(goals_file)

    try:
        goals_descriptions = _read_excel_file(goals_descriptions_file)
    except Exception as e:
        goals_descriptions = "Описания целей недоступны"
        print(f"Warning: Could not load goals descriptions: {e}")
    
    prompt = PROMPT_TEMPLATE_GOALS.format(
        goals_data=goals_json_data,
        goals_descriptions=goals_descriptions
    )

    print("Running LLM goals analysis...")
    result = await Runner.run(agent, input=prompt, run_config=rc)
    return result.final_output

async def run_analysis_simple(file_a: str, file_b: str):
    json_a = _load_parquet_summary_direct(file_a)
    json_b = _load_parquet_summary_direct(file_b)

    prompt = PROMPT_TEMPLATE_METRICS.format(
        version_a=json_a,
        version_b=json_b
    )

    result = await Runner.run(agent, input=prompt, run_config=rc)
    return result.final_output

if __name__ == "__main__":
    metrics_file = "data/full_metrics.parquet"
    
    print("UX metrics analysis started")
    
    out_metrics = asyncio.run(run_metrics_analysis(metrics_file))

    try:
        parsed_metrics = json.loads(out_metrics)
        with open("ux_metrics_analysis.json", "w", encoding="utf8") as f:
            json.dump(parsed_metrics, f, ensure_ascii=False, indent=2)
        
        analysis_metrics = parsed_metrics.get("analysis", [])
        total_metrics = len(analysis_metrics)
                    
    except json.JSONDecodeError:
        print("Error: Invalid response from LLM for metrics analysis")

    goals_file = "data/goal_stats_common_v1_v2.parquet"
    goals_descriptions_file = "data/Цели ЯМетрика.xlsx"
    
    print("\nUX goals analysis started")
    
    out_goals = asyncio.run(run_goals_analysis(goals_file, goals_descriptions_file))

    try:
        parsed_goals = json.loads(out_goals)
        with open("ux_goals_analysis.json", "w", encoding="utf8") as f:
            json.dump(parsed_goals, f, ensure_ascii=False, indent=2)
        
        analysis_goals = parsed_goals.get("analysis", [])
        total_goals = len(analysis_goals)
        
        print("Goals report saved to ux_goals_analysis.json")
                    
    except json.JSONDecodeError:
        print("Error: Invalid response from LLM for goals analysis")