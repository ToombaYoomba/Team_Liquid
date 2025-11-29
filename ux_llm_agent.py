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

REL_CHANGE_THRESHOLD = 40

PROMPT_TEMPLATE = """
Ты получаешь два набора данных (VERSION_A и VERSION_B) о поведении пользователей на сайте. 

ТВОЯ ЗАДАЧА: Найти ВСЕ метрики, которые изменились более чем на 40% между версиями, включая ВРЕМЕННЫЕ МЕТРИКИ.

ОБЯЗАТЕЛЬНЫЕ ШАГИ АНАЛИЗА:

1. Сначала проанализируй ОСНОВНЫЕ МЕТРИКИ СТРАНИЦ:
   - Просмотры страницы (views)
   - Доля трафика (percentage)

2. ЗАТЕМ ПРОАНАЛИЗИРУЙ ВРЕМЕННЫЕ РАСПРЕДЕЛЕНИЯ для каждой страницы:
   - hourly_distribution: сравни КАЖДЫЙ час отдельно
   - daily_distribution: сравни КАЖДЫЙ день отдельно
   - Если изменение по конкретному часу/дню >40% - включи в отчет

3. ПРОАНАЛИЗИРУЙ NUMERIC_STATS если доступны:
   - time_on_page, page_time, session_duration (временные метрики)
   - scroll_depth, clicks (метрики вовлеченности)

ДАННЫЕ VERSION_A:
{version_a}

ДАННЫЕ VERSION_B:
{version_b}

КОНКРЕТНЫЕ ПРИМЕРЫ КАК АНАЛИЗИРОВАТЬ ВРЕМЕННЫЕ ДАННЫЕ:

Для страницы "https://priem.mai.ru/rating/":
- Сравни hourly_distribution[10] в A и B (активность в 10 часов)
- Сравни daily_distribution["Monday"] в A и B (активность в понедельник)
- Если hourly_distribution[14] в A=100, в B=180 -> изменение 80% -> ВКЛЮЧИТЬ

Для страницы "homepage":
- Сравни просмотры по часам: hourly_distribution[9], hourly_distribution[10] и т.д.
- Сравни активность по дням: daily_distribution["Tuesday"], daily_distribution["Wednesday"]

Формат ОБЯЗАТЕЛЕН:
{{
  "анализ_результатов": {{
    "общее_сообщение": "строка",
    "всего_значимых_изменений": число,
    "статус": "есть_значимые_изменения/нет_значимых_изменений"
  }},
  "значимые_изменения_по_страницам": [
    {{
      "страница": "реальный_URL_или_категория",
      "проблемные_метрики": [
        {{
          "метрика": "название_метрики", 
          "тип_метрики": "просмотры/время/активность/клики/доля",
          "версия_A": число,
          "версия_B": число, 
          "абсолютное_изменение": число,
          "относительное_изменение_%": число,
          "значимость": "высокая/средняя/низкая",
          "интерпретация": "объяснение"
        }}
      ]
    }}
  ]
}}

ПРИМЕРЫ ВРЕМЕННЫХ МЕТРИК ДЛЯ ВКЛЮЧЕНИЯ:

"активность_в_10_часов_утра" (из hourly_distribution)
"активность_в_понедельник" (из daily_distribution)
"пиковая_активность_часа_14"
"среднее_время_на_странице" (из numeric_stats)
"время_сессии_медиана"

НЕ ИГНОРИРУЙ эти данные если они есть:
- hourly_distribution: распределение по часам (0-23)
- daily_distribution: распределение по дням недели
- numeric_stats: статистики по числовым колонкам
- session_metrics: метрики сессий

ВАЖНО: Если в данных есть временные распределения - ОБЯЗАТЕЛЬНО их анализируй и включай значимые изменения!
"""

agent = Agent(
    name="UX_MCP_Analyzer",
    instructions="""Ты — аналитик UX. Строго следуй правилам:
1. Анализируй только реальные страницы (URL и категории)
2. Включай только метрики с изменением >40%
3. Анализируй ВСЕ доступные данные включая временные распределения
4. Для hourly_distribution сравнивай каждый час отдельно
5. Для daily_distribution сравнивай каждый день отдельно
6. Возвращай только JSON в указанном формате""",
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
    Основная функция анализа с двумя файлами на версию
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
    
    print("Проверка доступных данных:")
    
    # Проверяем какие данные действительно доступны
    if 'url_category_analysis' in data_a['hits_analysis']:
        print("✓ Категории URL доступны")
    if 'top_urls' in data_a['hits_analysis']:
        print("✓ Топ URL доступны") 
    if 'session_metrics' in data_a['visits_analysis']:
        print("✓ Метрики сессий доступны")
    if 'time_range' in data_a['hits_analysis']:
        print("✓ Временные диапазоны доступны")
    if 'numeric_stats' in data_a['hits_analysis']:
        print("✓ Числовые статистики доступны")

    # Детальная проверка временных данных
    temporal_data_found = False
    
    if 'hourly_distribution' in data_a['hits_analysis']:
        hours_a = data_a['hits_analysis']['hourly_distribution']
        hours_b = data_b['hits_analysis']['hourly_distribution']
        print(f"✓ hourly_distribution доступен: {len(hours_a)} часов в A, {len(hours_b)} часов в B")
        print(f"  Пример часов в A: {dict(list(hours_a.items())[:5])}")  # первые 5 часов
        temporal_data_found = True
        
    if 'daily_distribution' in data_a['hits_analysis']:
        days_a = data_a['hits_analysis']['daily_distribution']
        days_b = data_b['hits_analysis']['daily_distribution']
        print(f"✓ daily_distribution доступен: {len(days_a)} дней в A, {len(days_b)} дней в B")
        print(f"  Дни в A: {days_a}")
        temporal_data_found = True

    if 'numeric_stats' in data_a['hits_analysis']:
        stats_a = data_a['hits_analysis']['numeric_stats']
        stats_b = data_b['hits_analysis']['numeric_stats']
        print(f"✓ numeric_stats доступен: {list(stats_a.keys())}")
        # Проверяем есть ли временные метрики в numeric_stats
        time_metrics = [k for k in stats_a.keys() if 'time' in k.lower() or 'duration' in k.lower()]
        if time_metrics:
            print(f"  Временные метрики в numeric_stats: {time_metrics}")
            temporal_data_found = True

    if not temporal_data_found:
        print("⚠️ Временные данные НЕ обнаружены в анализируемых данных")
    else:
        print("✅ Временные данные обнаружены - будут проанализированы")

    prompt = PROMPT_TEMPLATE.format(
        version_a=json_a,
        version_b=json_b
    )

    print("Запуск LLM анализа...")
    result = await Runner.run(agent, input=prompt, run_config=rc)
    return result.final_output

async def run_analysis_simple(file_a: str, file_b: str):
    """
    Старая функция для обратной совместимости
    """
    json_a = _load_parquet_summary_direct(file_a)
    json_b = _load_parquet_summary_direct(file_b)

    prompt = PROMPT_TEMPLATE.format(
        version_a=json_a,
        version_b=json_b
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
        
        # Анализ результата с новой структурой
        changes_by_page = parsed.get("значимые_изменения_по_страницам", [])
        total_changes = 0
        temporal_changes = 0
        
        if changes_by_page:
            print("\n" + "="*60)
            print("РЕЗУЛЬТАТЫ АНАЛИЗА:")
            print("="*60)
            
            for page_change in changes_by_page:
                page_name = page_change.get('страница', 'N/A')
                metrics = page_change.get('проблемные_метрики', [])
                total_changes += len(metrics)
                
                # Считаем временные метрики
                page_temporal = sum(1 for m in metrics if any(word in m.get('тип_метрики', '').lower() 
                                                            for word in ['время', 'активность', 'duration', 'time']))
                temporal_changes += page_temporal
                
                print(f"\n📄 {page_name}")
                print("-" * len(page_name))
                
                for metric in metrics:
                    metric_name = metric.get('метрика', 'N/A')
                    metric_type = metric.get('тип_метрики', 'N/A')
                    version_a = metric.get('версия_A', 0)
                    version_b = metric.get('версия_B', 0)
                    change_pct = metric.get('относительное_изменение_%', 0)
                    significance = metric.get('значимость', 'N/A')
                    interpretation = metric.get('интерпретация', 'N/A')
                    
                    # Разные иконки для разных типов метрик
                    icon = "📊"
                    if any(word in metric_type.lower() for word in ['время', 'активность', 'duration', 'time']):
                        icon = "⏰"
                        metric_type = "⏰ " + metric_type
                    elif any(word in metric_type.lower() for word in ['клики', 'clicks', 'click']):
                        icon = "🖱️"
                    elif any(word in metric_type.lower() for word in ['просмотры', 'views']):
                        icon = "📈"
                    
                    print(f"   {icon} {metric_name} ({metric_type}):")
                    print(f"      A: {version_a} → B: {version_b}")
                    print(f"      Изменение: {change_pct:+.1f}% ({significance})")
                    print(f"      💡 {interpretation}")
            
            print(f"\n✅ ИТОГО: {total_changes} значимых изменений (>40%) на {len(changes_by_page)} страницах")
            print(f"⏰ Временные метрики: {temporal_changes} изменений")
            
            # Статистика по типам метрик
            metric_types = {}
            for page_change in changes_by_page:
                for metric in page_change.get('проблемные_метрики', []):
                    m_type = metric.get('тип_метрики', 'другое')
                    metric_types[m_type] = metric_types.get(m_type, 0) + 1
            
            if metric_types:
                print("\n📈 РАСПРЕДЕЛЕНИЕ ПО ТИПАМ МЕТРИК:")
                for m_type, count in metric_types.items():
                    icon = "📊"
                    if any(word in m_type.lower() for word in ['время', 'активность']):
                        icon = "⏰"
                    elif any(word in m_type.lower() for word in ['клики']):
                        icon = "🖱️"
                    print(f"   {icon} {m_type}: {count} изменений")
                    
        else:
            print("❌ Значимых изменений не обнаружено (>40%)")
            print("   Проверьте отладочные файлы debug_version_*.json")
        
        print("\n📄 Отчет сохранен в ux_report_llm_significant.json")
                    
    except json.JSONDecodeError:
        print("❌ Ошибка: некорректный ответ от LLM")
        print("Ответ:", out)