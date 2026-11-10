import streamlit as st
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы дашборда
st.set_page_config(page_title="NBA Analytics ML Dashboard", layout="wide")
st.title("🏀 Аналитическая система NBA на базе Машинного Обучения")

# СТРОКА ПОДКЛЮЧЕНИЯ (ОБНОВЛЕННАЯ ПОД ВАШ СКУСШОТ)
DB_URI="postgresql://postgres.dcvbbfojssgfkobqjvkw:R4QAZsRLM2bjxk8J@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

@st.cache_data
def load_data_from_db():
    conn = psycopg2.connect(DB_URI)
    query = 'SELECT * FROM nba_players_stats;'
    df = pd.read_sql(query, conn)
    conn.close()
    return df

try:
    df = load_data_from_db()
    
    st.sidebar.header("Фильтры данных")
    all_players = sorted(df['Player'].unique())
    selected_player = st.sidebar.selectbox("Выберите игрока для анализа результативности:", all_players)
    
    cluster_mapping = {
        0: "Разыгрывающие / Плеймейкеры",
        1: "Универсальные форварды",
        2: "Атакующие защитники",
        3: "Центровые / Защитники кольца",
        4: "Звездные бомбардиры (Скореры)"
    }
    
    selected_cluster = st.sidebar.selectbox(
        "Фильтр по стилям игры (ML Кластеры):", 
        options=list(cluster_mapping.keys()), 
        format_func=lambda x: cluster_mapping[x]
    )

    tab1, tab2 = st.tabs(["📊 Прогноз результативности", "🤖 Кластеризация стилей игры"])
    
    with tab1:
        st.subheader(f"Анализ и предсказание очков для: {selected_player}")
        player_df = df[df['Player'] == selected_player].sort_values(by='Year')
        
        if not player_df.empty:
            st.dataframe(player_df[['Year', 'Tm', 'Age', 'G', 'MP', 'PTS', 'Predicted_PTS']])
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(player_df['Year'], player_df['PTS'], marker='o', label='Реальные очки (PTS)', color='blue', linewidth=2)
            ax.plot(player_df['Year'], player_df['Predicted_PTS'], marker='s', linestyle='--', label='Прогноз модели ML', color='orange', linewidth=2)
            ax.set_xlabel("Год")
            ax.set_ylabel("Очки за сезон")
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("Данные по игроку отсутствуют.")
            
    with tab2:
        st.subheader(f"Список игроков в группе: {cluster_mapping[selected_cluster]}")
        cluster_df = df[df['Cluster'] == selected_cluster]
        st.write(f"Всего игроков найдено в этом стиле: {len(cluster_df)}")
        st.dataframe(cluster_df[['Player', 'Year', 'Pos', 'PTS_per36', 'AST_per36', 'TRB_per36']].head(100))

except Exception as e:
    st.error(f"Ошибка подключения к базе данных Supabase: {e}")
    st.info("Пожалуйста, убедитесь, что вы верно указали пароль в файле app.py.")
