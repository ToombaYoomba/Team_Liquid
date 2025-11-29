import streamlit as st
import pandas as pd
import json
import os

def load_json_data(*args):
    with open(r"c:\Users\Екатерина\хакатон\ux_report_llm.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data


json_data = load_json_data('ux_report_llm.json')

st.set_page_config(page_title="Анализ UX по страницам", layout="wide")


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

st.title('Анализ UX метрик по страницам')
st.markdown('---')

def group_data_by_page(data):
    pages = {}
    for item in data['analysis']:
        page_name = item['page']
        if page_name not in pages:
            pages[page_name] = []
        pages[page_name].append(item)
    return pages


def calculate_page_metrics(pages_data):
    total_pages = len(pages_data)
    total_metrics = sum(len(metrics) for metrics in pages_data.values())
    critical_issues = 0
    significant_changes = 0
    for page_name, metrics_list in pages_data.items():
        for metric_data in metrics_list:
            change_percent = abs(100 - ((metric_data['version_b'] * 100) / metric_data['version_a']))
            if change_percent > 100:
                critical_issues += 1
            elif change_percent >= 50:
                significant_changes += 1

    
    return {
        'total_pages': total_pages,
        'total_metrics': total_metrics,
        'significant_changes': significant_changes,
        'critical_issues': critical_issues,
    }

pages_data = group_data_by_page(json_data)
overall_metrics = calculate_page_metrics(pages_data)

st.subheader('Общая статистика по страницам')

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Анализируемые страницы", overall_metrics['total_pages'])

with col2:
    st.metric("Всего метрик", overall_metrics['total_metrics'])

with col3:
    st.metric("Значимых изменений", overall_metrics['significant_changes'])

with col4:
    st.metric("Критических изменений", overall_metrics['critical_issues'])

st.markdown('---')


st.subheader('Детальный анализ по страницам')
st.markdown('---')
for page_name, metrics_list in pages_data.items():
    st.markdown(f"### {page_name.capitalize()}")
    
    
    page_critical = sum(1 for m in metrics_list if m['significant'] and m['relative_change'] > 1.5)
    page_improvements = sum(1 for m in metrics_list if m['significant'] and m['relative_change'] < 0.9)
    
    col_page1, col_page2, col_page3 = st.columns(3)
    
    with col_page1:
        st.metric("Метрик на странице", len(metrics_list))
    
    with col_page2:
        st.metric("Критических изменений", page_critical)
    st.markdown('---')
    
    

    for metric_data in metrics_list:
        metric_name = metric_data['metric']
        change_percent = abs(100 - ((metric_data['version_b'] * 100) / metric_data['version_a']))
        
 
        is_problem = change_percent > 100 and metric_data['significant']
        is_critical = change_percent >= 50 and metric_data['significant']
        is_improvement = change_percent < 50 and metric_data['significant']
        
        
        st.markdown(f"#### {metric_name.replace('_', ' ').title()}")
        
        
        col_compare1, col_compare2, col_compare3 = st.columns(3)
        
        with col_compare1:
            st.metric(
                "Версия A", 
                f"{metric_data['version_a']:.2f}",
                help="Исходное значение в версии A"
            )
        
        with col_compare2:
            st.metric(
                "Версия B", 
                f"{metric_data['version_b']:.2f}",
                help="Новое значение в версии B"
            )
        
        with col_compare3:
            delta_color = "normal" if change_percent <= 0 else "inverse"
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
        
        st.markdown("---")
    

    if len(metrics_list) > 1:
        st.markdown("**Сравнение всех метрик страницы:**")
        

        metrics_names = [m['metric'].replace('_', ' ').title() for m in metrics_list]
        values_a = [m['version_a'] for m in metrics_list]
        values_b = [m['version_b'] for m in metrics_list]
        

        chart_data = pd.DataFrame({
            'Метрики': metrics_names,
            'Версия A': values_a,
            'Версия B': values_b
        }).set_index('Метрики')
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.write(f"**Метрики {page_name} - Версия A**")
            st.bar_chart(chart_data['Версия A'])
        
        with col_chart2:
            st.write(f"**Метрики {page_name} - Версия B**")
            st.bar_chart(chart_data['Версия B'])
        
        st.markdown("---")

st.subheader('Сводная таблица по страницам')

table_data = []
for page_name, metrics_list in pages_data.items():
    for metric_data in metrics_list:
        change_percent = abs(100 - ((metric_data['version_b'] * 100) / metric_data['version_a']))
        
        is_problem = change_percent > 100 and metric_data['significant']
        is_critical = change_percent >= 50 and metric_data['significant']
        is_improvement = change_percent < 50 and metric_data['significant']
        

        if change_percent > 100:
            status = "Критично"
        elif change_percent >= 50:
            status = "Серьезно"
        else:
            status = "Некритично"
        
        table_data.append({
            "Страница": page_name,
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
