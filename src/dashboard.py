import streamlit as st
import pandas as pd
import json

def load_json_data(*args):
    with open("ux_metrics_analysis.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

json_data = load_json_data('ux_metrics_analysis.json')

st.set_page_config(page_title="Анализ метрик", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #DCDCDC;
    }
    .critical-alert {
        background-color: #FFDBDF;
        border: 1px solid #FFDBDF;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #252525;
    }
    .problem-alert {
        background-color: #FBD89F;
        border: 1px solid #FBD89F;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #252525;
    }
    .ok-alert {
        background-color: #DCDCDC;
        border: 1px solid #DCDCDC;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        color: #252525;
    }
</style>
""", unsafe_allow_html=True)

st.title('Анализ UX метрик')
st.markdown('---')

def calculate_overall_metrics(data):
    total_metrics = len(data['analysis'])
    critical_issues = 0
    significant_changes = 0
    
    for metric_data in data['analysis']:
        change_percent = abs(100 - ((metric_data['version_b'] * 100) / metric_data['version_a']))
        if change_percent > 70:
            critical_issues += 1
        elif change_percent >= 30:
            significant_changes += 1
    
    return {
        'total_metrics': total_metrics,
        'significant_changes': significant_changes,
        'critical_issues': critical_issues,
    }

overall_metrics = calculate_overall_metrics(json_data)

st.subheader('Общая статистика')

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Всего метрик", overall_metrics['total_metrics'])

with col2:
    st.metric("Значимых изменений", overall_metrics['significant_changes'])

with col3:
    st.metric("Критических изменений", overall_metrics['critical_issues'])

st.markdown('---')

st.subheader('Детальный анализ метрик')
st.markdown('---')


for i, metric_data in enumerate(json_data['analysis']):
    metric_name = metric_data['metric']
    change_percent = abs(100 - ((metric_data['version_b'] * 100) / metric_data['version_a']))
    
    is_critical = change_percent > 70 
    is_problem = change_percent >= 30 
    is_improvement = change_percent < 30 
    
    st.markdown(f"#### {metric_name.replace('_', ' ').title()}")
    

    col_compare1, col_compare2, col_compare3 = st.columns(3)
    
    with col_compare1:
        st.metric(
            "Версия A", 
            f"{metric_data['version_a']:.2f} {metric_data['unit']}",
            help="Исходное значение в версии A"
        )
    
    with col_compare2:
        st.metric(
            "Версия B", 
            f"{metric_data['version_b']:.2f} {metric_data['unit']}",
            help="Новое значение в версии B"
        )
    
    with col_compare3:
        if metric_data['version_b'] > metric_data['version_a']:
            sign = "+"
        else:
            sign = "-"
        st.metric(
            "Изменение", 
            f"{sign}{change_percent:.1f}%",
            delta=f"{sign}{change_percent:.1f}%"
        )
    

    if is_critical:
        st.markdown(f"""
        <div class="critical-alert">
            <strong>КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ:</strong> Показатель изменился на {change_percent:+.1f}%
        </div>
        """, unsafe_allow_html=True)
    elif is_problem:
        st.markdown(f"""
        <div class="problem-alert">
            <strong>СЕРЬЁЗНОЕ ИЗМЕНЕНИЕ:</strong> Показатель изменился на {change_percent:+.1f}%
        </div>
        """, unsafe_allow_html=True)
    elif is_improvement:
        st.markdown(f"""
        <div class="ok-alert">
            <strong>НЕБОЛЬШОЕ ИЗМЕНЕНИЕ:</strong> Показатель изменился на {change_percent:+.1f}%
        </div>
        """, unsafe_allow_html=True)
    

    st.markdown("**Анализ и рекомендации:**")
    st.info(metric_data['insight'])

    st.markdown("**Рекомендации:**")
    st.info(metric_data['solution'])
    
    st.markdown("---")


st.subheader('Сравнение всех метрик')


metrics_names = [f"{m['metric'].replace('_', ' ').title()}" for m in json_data['analysis']]
values_a = [m['version_a'] for m in json_data['analysis']]
values_b = [m['version_b'] for m in json_data['analysis']]


chart_data = pd.DataFrame({
    'Метрики': metrics_names,
    'Версия A': values_a,
    'Версия B': values_b
}).set_index('Метрики')

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.write("**Все метрики - Версия A**")
    st.bar_chart(chart_data['Версия A'])

with col_chart2:
    st.write("**Все метрики - Версия B**")
    st.bar_chart(chart_data['Версия B'])

st.markdown('---')


st.subheader('Сводная таблица всех метрик')

table_data = []
for metric_data in json_data['analysis']:
    change_percent = abs(100 - ((metric_data['version_b'] * 100) / metric_data['version_a']))
    

    if change_percent > 70:
        status = "Критично"
    elif change_percent >= 30:
        status = "Серьезно"
    else:
        status = "Некритично"
    
    table_data.append({
        "Метрика": metric_data['metric'],
        "Версия A": f"{metric_data['version_a']:.2f}",
        "Версия B": f"{metric_data['version_b']:.2f}",
        "Изменение %": f"{change_percent:+.1f}%",
        "Статус": status
    })

st.table(table_data)

st.markdown('---')

with st.expander("Просмотр исходных данных"):
    st.json(json_data)


