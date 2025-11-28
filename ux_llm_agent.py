import os
import json
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

from minimal_agents.agent import Agent
from minimal_agents.runner import Runner
from minimal_agents.run_config import RunConfig
from util.adk_custom_model_provider import CustomModelProvider

from mcp_ux_server import _load_parquet_summary_direct

# Загружаем переменные окружения
load_dotenv()
folder_id = os.environ["folder_id"]
api_key = os.environ["api_key"]

model = f"gpt://{folder_id}/qwen3-235b-a22b-fp8/latest"

client = AsyncOpenAI(
    base_url="https://rest-assistant.api.cloud.yandex.net/v1",
    api_key=api_key,
    project=folder_id
)

REL_CHANGE_THRESHOLD = 0.3
Z_SCORE_THRESHOLD = 2.0

PROMPT_TEMPLATE = """
У тебя есть две JSON-сводки — VERSION_A и VERSION_B.

VERSION_A:
{version_a}

VERSION_B:
{version_b}

Порог relative_change = {rel_thr}, z_threshold = {z_thr}.
Генерируй строго JSON.
"""

agent = Agent(
    name="UX_MCP_Analyzer",
    instructions="Ты — аналитик UX, выполняй анализ строго по инструкции и возвращай только JSON.",
    model=model
)

rc = RunConfig(model_provider=CustomModelProvider(model, client))


async def run_analysis(file_a: str, file_b: str):
    """
    Основная функция анализа:
    - Загружает parquet-файлы как JSON
    - Формирует промпт для LLM
    - Запускает агента
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
    a = "ux_version_A.parquet"
    b = "ux_version_B.parquet"
    out = asyncio.run(run_analysis(a, b))

    try:
        parsed = json.loads(out)
        with open("ux_report_llm.json", "w", encoding="utf8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        print("Saved ux_report_llm.json")
    except json.JSONDecodeError:
        print("Invalid JSON from LLM:")
        print(out)
