# =============================================================================
# SISTEMA DE PREDICCIÓN DE DESERCIÓN ESTUDIANTIL
# =============================================================================
# Descripción: Aplicación web interactiva para predecir deserción estudiantil
#              utilizando Machine Learning (KNN + SMOTE)
# Metodología: CRISP-DM
# Autor: [Tu Nombre]
# Fecha: Febrero 2026
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go

# ========================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS
# ========================================
st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado con colores claros y legibles
st.markdown("""
<style>
    /* Importar fuente */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fondo de la app */
    .stApp {
        background-color: #0f172a;
    }
    
    /* Contenedor principal - AZUL OSCURO */
    .main .block-container {
        padding: 2rem;
        background-color: #1e3a5f;
        border-radius: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        max-width: 1400px;
        margin: 2rem auto;
    }
    
    /* Títulos principales */
    h1 {
        color: #ffffff !important;
        font-weight: 800;
        text-align: center;
        font-size: 2.8rem !important;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #ffffff !important;
        font-weight: 700;
        border-left: 5px solid #3b82f6;
        padding-left: 15px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    h4 {
        color: #ffffff !important;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #cbd5e1;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #f1f5f9 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #f1f5f9 !important;
        font-weight: 600;
        font-size: 1.05rem;
    }
    
    /* Botones */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        font-weight: 700;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0f172a;
        padding: 10px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #334155;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        color: #cbd5e1;
        border: 2px solid #475569;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white !important;
        border-color: #3b82f6;
    }
    
    /* Cards personalizadas */
    .custom-card {
        background: #2d4a6f;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        border-left: 5px solid #3b82f6;
        margin-bottom: 1rem;
    }
    
    /* Divisores */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #475569, transparent);
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        border-color: #cbd5e1;
        color: #1e293b;
        background-color: white;
    }
    
    /* Alertas */
    .stAlert {
        border-radius: 12px;
    }
    
    /* Asegurar texto visible en dataframes */
    .stDataFrame {
        color: #1e293b;
    }
    
    /* Texto de párrafos y captions */
    p {
        color: #e2e8f0;
    }
    
    /* Captions */
    .stCaptionContainer {
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# FUNCIONES DE CARGA Y ENTRENAMIENTO
# ========================================

@st.cache_data
def cargar_datos():
    """
    Carga y prepara el dataset de estudiantes.
    
    Returns:
        pd.DataFrame: DataFrame con datos limpios y variable objetivo definida
    """
    # Leer CSV con configuración específica para manejar formato
    df = pd.read_csv(
        "REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv",
        sep=";",
        encoding="latin1",
        on_bad_lines="skip"
    )
    
    # Detectar columnas de promedio y asistencia dinámicamente
    promedio_col = [c for c in df.columns if 'PROMEDIO' in c.upper()][0]
    asistencia_col = [c for c in df.columns if 'ASISTENCIA' in c.upper()][0]
    
    # Limpiar y convertir datos numéricos
    for col in [promedio_col, asistencia_col]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "."),
            errors='coerce'
        )
    
    # Eliminar registros con valores faltantes
    df = df.dropna(subset=[promedio_col, asistencia_col])
    
    # Renombrar columnas para facilitar el procesamiento
    df = df.rename(columns={
        promedio_col: 'PROMEDIO',
        asistencia_col: 'ASISTENCIA'
    })
    
    # DEFINIR VARIABLE OBJETIVO DE DESERCIÓN
    # Criterio: Promedio < 7.0 o Asistencia < 70%
    df["DESERCION"] = (
         (df["PROMEDIO"] < 7) | (df["ASISTENCIA"] < 70)
    ).astype(int)
    
    return df

@st.cache_resource
def entrenar_modelo(df):
    """
    Entrena el modelo KNN con balanceo SMOTE.
    
    Args:
        df: DataFrame con datos de estudiantes
    
    Returns:
        tuple: (modelo entrenado, scaler, métricas de evaluación)
    """
    
    # Seleccionar features (X) y target (y)
    X = df[["PROMEDIO", "ASISTENCIA"]]
    y = df["DESERCION"]
    
    # División estratificada train/test (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalización de features con StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Aplicar SMOTE para balancear clases
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Entrenamiento del modelo KNN
    model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predicciones en conjunto de prueba
    y_pred = model.predict(X_test_scaled)
    
    # Calcular métricas
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metricas = {
        'confusion_matrix': cm,
        'report': report,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test
    }
    
    return model, scaler, metricas

# ========================================
# CARGAR DATOS Y MODELO
# ========================================
df = cargar_datos()
model, scaler, metricas = entrenar_modelo(df)

# ========================================
# SIDEBAR - NAVEGACIÓN
# ========================================
with st.sidebar:
    st.markdown("<div style='text-align: center; padding: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='font-size: 3rem; margin: 0;'>👤</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size: 1.3rem; margin: 0.5rem 0;'> Sistema de Predicción</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<hr style='background: rgba(255,255,255,0.3); margin: 1rem 0;'>", unsafe_allow_html=True)
    
    opcion = st.radio(
        "Navegación",
        ["💠 Inicio", "📊 Análisis y Métricas", "🔮 Predicción Individual"],
    )
    
    st.markdown("<hr style='background: rgba(255,255,255,0.3); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
        <h4 style='margin-top: 0; font-size: 1rem;'>📊 Información</h4>
        <p style='margin: 5px 0; font-size: 0.9rem;'><b>Modelo:</b> KNN</p>
        <p style='margin: 5px 0; font-size: 0.9rem;'><b>Técnica:</b> SMOTE</p>
        <p style='margin: 5px 0; font-size: 0.9rem;'><b>Variables:</b> Promedio y Asistencia</p>
    </div>
    """, unsafe_allow_html=True)

# ========================================
# PÁGINA: INICIO
# ========================================
if opcion == "💠 Inicio":
    st.markdown("<h1>👤 Sistema de Predicción de Deserción Estudiantil</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #cbd5e1; margin-bottom: 2rem;</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    en_riesgo = df["DESERCION"].sum()
    sin_riesgo = len(df) - en_riesgo
    
    with col1:
        st.markdown("""
        <div class='custom-card' style='text-align: center;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>👥</div>
            <div style='font-size: 2rem; font-weight: 800; color: #3b82f6;'>{:,}</div>
            <div style='font-size: 0.95rem; color: #000000; font-weight: 600; margin-top: 0.5rem;'>Total Estudiantes</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='custom-card' style='text-align: center; border-left-color: #ef4444;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>⚠️</div>
            <div style='font-size: 2rem; font-weight: 800; color: #ef4444;'>{:,}</div>
            <div style='font-size: 0.95rem; color: #000000; font-weight: 600; margin-top: 0.5rem;'>En Riesgo ({:.1f}%)</div>
        </div>
        """.format(en_riesgo, (en_riesgo/len(df)*100)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='custom-card' style='text-align: center; border-left-color: #10b981;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>✅</div>
            <div style='font-size: 2rem; font-weight: 800; color: #10b981;'>{:,}</div>
            <div style='font-size: 0.95rem; color: #000000; font-weight: 600; margin-top: 0.5rem;'>Sin Riesgo ({:.1f}%)</div>
        </div>
        """.format(sin_riesgo, (sin_riesgo/len(df)*100)), unsafe_allow_html=True)
    
    with col4:
        cm = metricas['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        st.markdown("""
        <div class='custom-card' style='text-align: center; border-left-color: #f59e0b;'>
            <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🎯</div>
            <div style='font-size: 2rem; font-weight: 800; color: #f59e0b;'>{:.1f}%</div>
            <div style='font-size: 0.95rem; color: #000000; font-weight: 600; margin-top: 0.5rem;'>Recall del Modelo</div>
        </div>
        """.format(recall * 100), unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Descripción del sistema
    st.markdown("""
    <div class='custom-card'>
        <h2 style='margin-top: 0; color: #000000;'>📋 Acerca del Sistema</h2>
        <p style='font-size: 1.05rem; line-height: 1.8; color: #e2e8f0;'>
            Este sistema utiliza <b>Machine Learning</b> para predecir el riesgo de deserción 
            estudiantil basándose en dos variables fundamentales:
        </p>
        <ul style='font-size: 1rem; line-height: 2; color: #e2e8f0;'>
            <li><b>📚 Promedio académico:</b> Rendimiento general del estudiante</li>
            <li><b>✅ Asistencia:</b> Porcentaje de asistencia a clases</li>
        </ul>
        <p style='font-size: 1.05rem; line-height: 1.8; color: #e2e8f0; margin-top: 1rem;'>
            El modelo emplea el algoritmo <b>K-Nearest Neighbors (KNN)</b> con técnicas 
            de balanceo <b>SMOTE</b> para identificar estudiantes en riesgo y facilitar 
            intervenciones tempranas.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Gráfico de distribución general
    st.markdown("<h2 style='text-align: center;'>📊 Distribución General de Riesgo</h2>", unsafe_allow_html=True)
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Sin Riesgo', 'En Riesgo'],
            y=[sin_riesgo, en_riesgo],
            marker=dict(
                color=['#10b981', '#ef4444'],
                line=dict(color='white', width=2)
            ),
            text=[f'{sin_riesgo:,}<br>({sin_riesgo/len(df)*100:.1f}%)', 
                  f'{en_riesgo:,}<br>({en_riesgo/len(df)*100:.1f}%)'],
            textposition='auto',
            textfont=dict(size=14, color='white', family='Arial'),
            hovertemplate='<b>%{x}</b><br>Estudiantes: %{y:,}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        xaxis_title='Categoría',
        yaxis_title='Cantidad de Estudiantes',
        plot_bgcolor='#2d4a6f',
        paper_bgcolor='#2d4a6f',
        height=400,
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(color='#e2e8f0')
    )
    
    st.plotly_chart(fig, width="stretch")

# ========================================
# PÁGINA: ANÁLISIS Y MÉTRICAS
# ========================================
elif opcion == "📊 Análisis y Métricas":
    st.markdown("<h1>📊 Análisis de Datos y Evaluación del Modelo</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ===== SECCIÓN 1: ANÁLISIS EXPLORATORIO =====
    st.markdown("<h2>📈 Análisis Exploratorio de Datos</h2>", unsafe_allow_html=True)
    
    # Estadísticas descriptivas
    st.markdown("<h3>Estadísticas Descriptivas</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.dataframe(
        df[["PROMEDIO", "ASISTENCIA"]].describe().round(2),
        width="stretch"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visualizaciones - Distribuciones
    st.markdown("<h3>Distribuciones</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, 
            x="PROMEDIO", 
            nbins=30,
            title="Distribución del Promedio Académico",
            color_discrete_sequence=['#3b82f6']
        )
        fig.add_vline(
            x=df["PROMEDIO"].mean(), 
            line_dash="dash", 
            line_color="#fbbf24",
            annotation_text=f"Media: {df['PROMEDIO'].mean():.2f}"
        )
        fig.update_layout(
            xaxis_title="Promedio",
            yaxis_title="Frecuencia",
            height=400,
            plot_bgcolor='#2d4a6f',
            paper_bgcolor='#2d4a6f',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        fig = px.histogram(
            df, 
            x="ASISTENCIA", 
            nbins=30,
            title="Distribución de Asistencia",
            color_discrete_sequence=['#10b981']
        )
        fig.add_vline(
            x=df["ASISTENCIA"].mean(), 
            line_dash="dash", 
            line_color="#fbbf24",
            annotation_text=f"Media: {df['ASISTENCIA'].mean():.1f}%"
        )
        fig.update_layout(
            xaxis_title="Asistencia (%)",
            yaxis_title="Frecuencia",
            height=400,
            plot_bgcolor='#2d4a6f',
            paper_bgcolor='#2d4a6f',
            font=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, width="stretch")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ===== SECCIÓN 2: MÉTRICAS DEL MODELO =====
    st.markdown("<h2>📈 Evaluación del Modelo</h2>", unsafe_allow_html=True)
    
    # Calcular métricas
    cm = metricas['confusion_matrix']
    report = metricas['report']
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    st.markdown("<h3>Métricas Principales</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("Accuracy", accuracy, "📊", "#3b82f6", "Aciertos totales"),
        ("Precision", precision, "🎯", "#10b981", "Predicciones correctas"),
        ("Recall", recall, "⭐", "#f59e0b", "Detección de riesgo"),
        ("F1-Score", f1, "⚖️", "#8b5cf6", "Balance general")
    ]
    
    for col, (nombre, valor, icono, color, desc) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            destacado = "border: 3px solid #f59e0b; box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);" if nombre == "Recall" else ""
            st.markdown(f"""
            <div class='custom-card' style='border-left-color: {color}; text-align: center; {destacado}'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icono}</div>
                <div style='font-size: 2rem; font-weight: 800; color: {color};'>{valor:.1%}</div>
                <div style='font-size: 1rem; color: #000000; font-weight: 700; margin: 8px 0;'>{nombre}</div>
                <div style='font-size: 0.85rem; color: #cbd5e1;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Matriz de confusión
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("<h3>Matriz de Confusión</h3>", unsafe_allow_html=True)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicho: Sin Riesgo', 'Predicho: En Riesgo'],
            y=['Real: Sin Riesgo', 'Real: En Riesgo'],
            colorscale=[[0, '#e0f2fe'], [0.5, '#7dd3fc'], [1, '#0284c7']],
            text=cm,
            texttemplate='<b>%{text}</b>',
            textfont={"size": 20, "color": "#1e293b"},
            showscale=False
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='#2d4a6f',
            paper_bgcolor='#2d4a6f',
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed'),
            font=dict(color='#e2e8f0')
        )
        
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        st.markdown("<h3>Interpretación</h3>", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='custom-card' style='border-left-color: #10b981;'>
            <h4 style='color: #000000; margin: 0;'>🟢 Verdaderos Negativos: {tn}</h4>
            <p style='margin: 5px 0 0 0; color: #cbd5e1;'>Sin riesgo predichos correctamente</p>
        </div>
        
        <div class='custom-card' style='border-left-color: #f59e0b;'>
            <h4 style='color: #000000; margin: 0;'>🟡 Falsos Positivos: {fp}</h4>
            <p style='margin: 5px 0 0 0; color: #cbd5e1;'>Falsa alarma</p>
        </div>
        
        <div class='custom-card' style='border-left-color: #ef4444;'>
            <h4 style='color: #000000; margin: 0;'>🔴 Falsos Negativos: {fn}</h4>
            <p style='margin: 5px 0 0 0; color: #cbd5e1;'>¡PELIGRO! No detectados</p>
        </div>
        
        <div class='custom-card' style='border-left-color: #10b981;'>
            <h4 style='color: #000000; margin: 0;'>🟢 Verdaderos Positivos: {tp}</h4>
            <p style='margin: 5px 0 0 0; color: #cbd5e1;'>En riesgo detectados</p>
        </div>
        """, unsafe_allow_html=True)

# ========================================
# PÁGINA: PREDICCIÓN 
# ========================================
elif opcion == "🔮 Predicción":
    st.markdown("<h1>🔮 Predicción de Riesgo </h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='custom-card' style='background-color: #3b82f6; border: none;'>
        <h3 style='color: white; margin-top: 0; text-align: center;'>Ingresa los datos del estudiante</h3>
        <p style='color: white; text-align: center; margin: 0;'>El sistema analizará el riesgo de deserción en tiempo real</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("####  Promedio Académico")
        promedio = st.number_input(
            "Promedio",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            help="Promedio académico del estudiante (0-10)",
            label_visibility="collapsed"
        )
        st.caption(f"Promedio general: {df['PROMEDIO'].mean():.2f}")
    
    with col2:
        st.markdown("####  Asistencia")
        asistencia = st.number_input(
            "Asistencia",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0,
            help="Porcentaje de asistencia",
            label_visibility="collapsed"
        )
        st.caption(f"Promedio general: {df['ASISTENCIA'].mean():.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predecir_btn = st.button("🔮 Predecir Riesgo", type="primary", width="stretch")
    
    if predecir_btn:
        # Preparar datos para predicción
        datos_estudiante = np.array([[promedio, asistencia]])
        datos_escalados = scaler.transform(datos_estudiante)
        
        # Realizar predicción
        prediccion = model.predict(datos_escalados)[0]
        probabilidades = model.predict_proba(datos_escalados)[0]
        prob_riesgo = probabilidades[1] * 100
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if prediccion == 1:
            # ESTUDIANTE EN RIESGO
            st.markdown(f"""
            <div class='custom-card' style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); border: none; text-align: center; padding: 2.5rem;'>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>⚠️</div>
                <h1 style='color: white; font-size: 2.2rem; margin: 0;'>ESTUDIANTE EN RIESGO</h1>
                <h2 style='color: white; font-size: 2.5rem; margin: 1rem 0;'>{prob_riesgo:.1f}%</h2>
                <p style='color: white; font-size: 1.1rem; margin: 0;'>Probabilidad de Deserción</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class='custom-card' style='border-left-color: #ef4444;'>
                    <h3 style='color: #000000; margin-top: 0;'>🚨 Acciones Inmediatas</h3>
                    <ul style='line-height: 2; color: #e2e8f0;'>
                        <li>Contacto urgente con el estudiante</li>
                        <li>Asignar tutor académico</li>
                        <li>Evaluación psicopedagógica</li>
                        <li>Revisar situación socioeconómica</li>
                        <li>Crear plan de intervención</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='custom-card' style='border-left-color: #f59e0b;'>
                    <h3 style='color: #000000; margin-top: 0;'>📋 Seguimiento</h3>
                    <ul style='line-height: 2; color: #e2e8f0;'>
                        <li>Reuniones semanales</li>
                        <li>Monitoreo de asistencia diaria</li>
                        <li>Apoyo en métodos de estudio</li>
                        <li>Servicios de apoyo</li>
                        <li>Evaluación mensual</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # ESTUDIANTE SIN RIESGO
            st.markdown(f"""
            <div class='custom-card' style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); border: none; text-align: center; padding: 2.5rem;'>
                <div style='font-size: 4rem; margin-bottom: 1rem;'>✅</div>
                <h1 style='color: white; font-size: 2.2rem; margin: 0;'>ESTUDIANTE SIN RIESGO</h1>
                <h2 style='color: white; font-size: 2.5rem; margin: 1rem 0;'>{prob_riesgo:.1f}%</h2>
                <p style='color: white; font-size: 1.1rem; margin: 0;'>Probabilidad de Deserción (Baja)</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='custom-card' style='border-left-color: #10b981;'>
                <h3 style='color: #000000; margin-top: 0;'>✨ Recomendaciones</h3>
                <ul style='line-height: 2; color: #e2e8f0;'>
                    <li>Seguimiento regular</li>
                    <li>Reforzar hábitos positivos</li>
                    <li>Mantener comunicación abierta</li>
                    <li>Monitorear rendimiento</li>
                    <li>Oportunidades de desarrollo</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Análisis comparativo
        st.markdown("<h2>📊 Análisis Comparativo</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            comparacion = pd.DataFrame({
                'Variable': ['Promedio', 'Asistencia (%)'],
                'Estudiante': [promedio, asistencia],
                'Promedio General': [
                    df['PROMEDIO'].mean(),
                    df['ASISTENCIA'].mean()
                ]
            })
            
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top: 0; color: #000000;'>Comparación</h3>", unsafe_allow_html=True)
            st.dataframe(comparacion.round(2), width="stretch", hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-top: 0; color: #000000;'>Factores</h3>", unsafe_allow_html=True)
            
            if promedio < 7:
                st.markdown("<p style='color: #e2e8f0;'>❌ Promedio bajo (< 7.0)</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #e2e8f0;'>✅ Promedio aceptable</p>", unsafe_allow_html=True)
            
            if asistencia < 70:
                st.markdown("<p style='color: #e2e8f0;'>❌ Asistencia baja (< 70%)</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #e2e8f0;'>✅ Asistencia aceptable</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Gráfico de importancia de variables
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2>📈 Importancia de Variables</h2>", unsafe_allow_html=True)
        
        import_promedio = abs(promedio - df['PROMEDIO'].mean()) / df['PROMEDIO'].std()
        import_asistencia = abs(asistencia - df['ASISTENCIA'].mean()) / df['ASISTENCIA'].std()
        
        total_import = import_promedio + import_asistencia
        
        if total_import > 0:
            importancias = {
                'Promedio': (import_promedio / total_import) * 100,
                'Asistencia': (import_asistencia / total_import) * 100
            }
        else:
            importancias = {'Promedio': 50.0, 'Asistencia': 50.0}
        
        fig = go.Figure(go.Bar(
            y=list(importancias.keys()),
            x=list(importancias.values()),
            orientation='h',
            marker=dict(color=['#3b82f6', '#10b981']),
            text=[f'{v:.1f}%' for v in importancias.values()],
            textposition='auto',
            textfont=dict(color='white')
        ))
        
        fig.update_layout(
            xaxis_title='Importancia Relativa (%)',
            height=300,
            plot_bgcolor='#2d4a6f',
            paper_bgcolor='#2d4a6f',
            font=dict(color='#e2e8f0')
        )
        
        st.plotly_chart(fig, width="stretch")
