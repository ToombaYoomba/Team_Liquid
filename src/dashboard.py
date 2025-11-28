import streamlit as st
import json
import os

def load_json_data(*args):
    with open(r"c:\Users\Екатерина\хакатон\ux_report.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data


json_data = load_json_data('ux_report.json')

st.set_page_config(page_title="Анализ UX проблем", layout="wide")
st.title('Анализ разницы между версиями')
st.markdown('---')

def calculate_overall_metrics(data):
    detected_problems = data['detected_problems']
    
    total_problems = len(detected_problems)
    total_pages_affected = 0
    critical_issues = 0
    improvements = 0
    
    for problem, pages in detected_problems.items():
        total_pages_affected += len(pages)
        for page, metrics in pages.items():
            metric_name = list(metrics.keys())[0]
            change_coef = metrics[metric_name]['change_coef']
            
            if change_coef > 2:  
                critical_issues += 1
            elif change_coef < 0.8: 
                improvements += 1
    
    return {
        'total_problems': total_problems,
        'total_pages_affected': total_pages_affected,
        'critical_issues': critical_issues,
        'improvements': improvements,
        'file_A': data['file_A'],
        'file_B': data['file_B']
    }


metrics = calculate_overall_metrics(json_data)
detected_problems = json_data['detected_problems']


st.subheader('Общая статистика проблем')

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Обнаружено проблем", metrics['total_problems'])

with col2:
    st.metric("Проанализировано изменений", metrics['total_pages_affected'])

with col3:
    st.metric("Улучшения", metrics['improvements'])

st.markdown('---')
st.subheader('Сравнение версий')

col5, col6 = st.columns(2)

with col5:
    st.info(f"**Версия A:** {metrics['file_A']}")
    
with col6:
    st.info(f"**Версия B:** {metrics['file_B']}")

st.markdown('---')
st.subheader('Детальный анализ проблем')

k = 0
for problem_name, pages_data in detected_problems.items():
    k += 1
    st.markdown(f"### {k}. {problem_name}")
    
    problem_pages = len(pages_data)
    avg_change = sum(data[list(data.keys())[0]]['change_coef'] for data in pages_data.values()) / problem_pages
    max_change = max(data[list(data.keys())[0]]['change_coef'] for data in pages_data.values())
    
    col7, col8, col9, col10 = st.columns(4)
    
    with col7:
        st.metric("Проанализировано изменений", problem_pages)
    
    with col8:
        st.metric("Среднее изменение", f"{avg_change:.2f}x")
    
    with col9:
        st.metric("Макс. изменение", f"{max_change:.2f}x")
    
    with col10:
        if max_change > 2:
            status = "Критическая"
        elif max_change > 1.5:
            status = "Серьёзная"
        else:
            status = "Умеренная"
        st.metric("Статус проблемы", status)
    
    st.markdown("**Детали по страницам:**")
    
    for page_name, metrics_data in pages_data.items():
        metric_type = list(metrics_data.keys())[0]
        metric_values = metrics_data[metric_type]
        
        col11, col12, col13, col14 = st.columns(4)
        
        with col11:
            st.metric(
                f"{page_name} - Версия A", 
                f"{metric_values['value_in_A']:.2f}",
                help=f"Метрика: {metric_type}"
            )
        
        with col12:
            st.metric(
                f"{page_name} - Версия B", 
                f"{metric_values['value_in_B']:.2f}",
                help=f"Метрика: {metric_type}"
            )
        
        with col13:
            change_percent = (metric_values['change_coef'] - 1) * 100
            delta_color = "normal" if change_percent <= 0 else "inverse"
            st.metric(
                f"{page_name} - Изменение", 
                f"{change_percent:+.1f}%",
                delta=f"{change_percent:+.1f}%",
                delta_color=delta_color
            )
        
        with col14:
            if metric_values['change_coef'] > 1:
                trend = "Ухудшение"
            else:
                trend = "Улучшение"
            st.metric("Тренд", trend)
    
    if len(pages_data) > 1:
        col15, col16 = st.columns(2)
        
        with col15:
            st.write(f"**Значения в версии A**")
            values_a = {page: data[list(data.keys())[0]]['value_in_A'] for page, data in pages_data.items()}
            st.bar_chart(values_a)
        
        with col16:
            st.write(f"**Значения в версии B**")
            values_b = {page: data[list(data.keys())[0]]['value_in_B'] for page, data in pages_data.items()}
            st.bar_chart(values_b)
    
    st.markdown("---")

st.subheader('Сводная таблица проблем')

table_data = []
for problem_name, pages_data in detected_problems.items():
    for page_name, metrics_data in pages_data.items():
        metric_type = list(metrics_data.keys())[0]
        metric_values = metrics_data[metric_type]
        
        change_percent = (metric_values['change_coef'] - 1) * 100
        
        table_data.append({
            "Проблема": problem_name,
            "Страница": page_name,
            "Метрика": metric_type,
            "Версия A": f"{metric_values['value_in_A']:.2f}",
            "Версия B": f"{metric_values['value_in_B']:.2f}",
            "Изменение": f"{change_percent:+.1f}%",
            "Статус": "Критично" if metric_values['change_coef'] > 2 else 
                     "Серьёзно" if metric_values['change_coef'] > 1.5 else 
                     "Умеренно"
        })

st.table(table_data)

st.markdown('---')
st.subheader('Анализ критических проблем')


critical_issues = []
for problem_name, pages_data in detected_problems.items():
    for page_name, metrics_data in pages_data.items():
        metric_values = metrics_data[list(metrics_data.keys())[0]]
        if metric_values['change_coef'] > 2:  
            critical_issues.append({
                'problem': problem_name,
                'page': page_name,
                'change_coef': metric_values['change_coef'],
                'metric': list(metrics_data.keys())[0]
            })


critical_issues.sort(key=lambda x: x['change_coef'], reverse=True)

if critical_issues:
    st.warning(f"**Обнаружено {len(critical_issues)} критических проблем:**")
    
    for i, issue in enumerate(critical_issues[:5], 1):  
        change_percent = (issue['change_coef'] - 1) * 100
        st.error(f"{i}. **{issue['problem']}** на странице **{issue['page']}** - ухудшение на {change_percent:+.1f}%")
else:
    st.success("Критических проблем не обнаружено!")


st.markdown('---')
st.subheader('Рекомендации по улучшению')

col17, col18, col19 = st.columns(3)

with col17:
    st.info("**Приоритетные зоны:**")
    if critical_issues:
        for issue in critical_issues[:3]:
            st.write(f"• {issue['page']} - {issue['problem']}")
    else:
        st.write(" Критических зон не обнаружено")

with col18:
    st.info("**Метрики для мониторинга:**")
    unique_metrics = set()
    for problem_data in detected_problems.values():
        for page_data in problem_data.values():
            unique_metrics.add(list(page_data.keys())[0])
    for metric in list(unique_metrics)[:3]:
        st.write(f"• {metric}")

with col19:
    st.info("**Следующие шаги:**")
    st.write("• Провести A/B тестирование")
    st.write("• Улучшить навигацию")
    st.write("• Оптимизировать формы ввода")

with st.expander("ℹОписание данных и метрик"):
    st.write("""
    **Структура данных:**
    - **file_A, file_B**: Файлы данных для сравнения версий
    - **detected_problems**: Обнаруженные проблемы UX
    
    **Типы метрик:**
    - **bounce_rate**: Показатель отскока
    - **time_on_page_avg**: Среднее время на странице
    - **pages_per_session**: Страниц за сессию
    - **conversions**: Конверсии
    - **form_abandoned**: Отказы от форм
    - **field_errors**: Ошибки в полях
    
    **Интерпретация коэффициента изменения:**
    - > 1.5x: Критическое ухудшение
    - 1.0-1.5x: Умеренное ухудшение  
    - < 1.0x: Улучшение показателя
    """)


with st.expander("Просмотр исходных JSON данных"):
    st.json(json_data)