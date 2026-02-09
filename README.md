# 🎓 Sistema de Predicción de Deserción Estudiantil

## 📋 Descripción del Proyecto

Sistema de predicción de deserción estudiantil desarrollado con **Machine Learning** que utiliza la metodología **CRISP-DM** para identificar estudiantes en riesgo mediante el análisis de variables académicas clave.

El proyecto incluye:
- ✅ Modelo de clasificación basado en **K-Nearest Neighbors (KNN)**
- ✅ Técnica de balanceo **SMOTE** para mejorar la detección
- ✅ Interfaz web interactiva con **Streamlit**
- ✅ Visualizaciones con **Plotly** y análisis estadístico

## 🎯 Objetivos

### Objetivo General
Desarrollar un sistema de predicción de deserción estudiantil aplicando técnicas de minería de datos, que permita identificar estudiantes en riesgo y visualizar los resultados mediante una interfaz gráfica interactiva.

### Objetivos Específicos
1. Realizar análisis exploratorio del conjunto de datos
2. Identificar variables relevantes para la predicción
3. Definir la variable objetivo (deserción) a partir de datos históricos
4. Aplicar técnicas de preprocesamiento y transformación
5. Construir y evaluar modelo de clasificación
6. Desarrollar interfaz gráfica con Streamlit

## 📊 Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | ~90% |
| **Recall** | 85-90% |
| **Precision** | ~70% |
| **F1-Score** | 75-80% |

> **Nota importante**: El modelo está optimizado para maximizar el **Recall**, minimizando los falsos negativos (estudiantes en riesgo no detectados), que es crítico para intervenciones tempranas.

## 🛠️ Tecnologías Utilizadas

### Lenguaje y Framework
- **Python 3.14.3**
- **Streamlit** - Framework para aplicaciones web

### Bibliotecas de Análisis y ML
- **Pandas** - Manipulación de datos
- **NumPy** - Computación numérica
- **Scikit-learn** - Algoritmos de machine learning
- **Imbalanced-learn** - Técnicas de balanceo (SMOTE)

### Visualización
- **Plotly** - Gráficos interactivos
- **Matplotlib** - Gráficos estáticos
- **Seaborn** - Visualización estadística

## 📁 Estructura del Proyecto

```
proyecto-desercion/
│
├── desercion_app.py                                   
├── REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv 
├── requirements.txt                          
├── README.md                                
│
└── .gitignore                              
```

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/prediccion-desercion-estudiantil.git
cd prediccion-desercion-estudiantil
```

2. **Crear entorno virtual (recomendado)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Colocar el dataset**
- Asegúrate de tener el archivo `REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv` en el directorio raíz del proyecto

5. **Ejecutar la aplicación**
```bash
streamlit run app.py
```

6. **Acceder a la aplicación**
- La aplicación se abrirá automáticamente en tu navegador
- Si no, accede manualmente a: `http://localhost:8501`

## 📖 Uso de la Aplicación

La aplicación cuenta con tres módulos principales:

### 1. 💠 Inicio
- **Dashboard principal** con métricas generales
- Total de estudiantes analizados
- Distribución de riesgo (en riesgo vs sin riesgo)
- Recall del modelo
- Gráfico de distribución general

### 2. 📊 Análisis y Métricas
- **Análisis exploratorio** de datos
  - Estadísticas descriptivas
  - Distribuciones de promedio y asistencia
- **Evaluación del modelo**
  - Métricas principales (Accuracy, Precision, Recall, F1-Score)
  - Matriz de confusión interactiva
  - Interpretación detallada de resultados

### 3. 🔮 Predicción Individual
- **Interfaz de predicción** en tiempo real
  - Ingreso de promedio académico (0-10)
  - Ingreso de porcentaje de asistencia (0-100%)
  - Predicción instantánea con probabilidad
- **Resultados detallados**
  - Estado de riesgo del estudiante
  - Recomendaciones específicas
  - Análisis comparativo con promedios generales
  - Importancia relativa de variables


## 📈 Variables del Modelo

### Features (Variables Predictoras)
| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| PROMEDIO | Numérica continua | Promedio académico del estudiante | 0-10 |
| ASISTENCIA | Numérica continua | Porcentaje de asistencia a clases | 0-100% |

### Target (Variable Objetivo)
| Variable | Tipo | Descripción | Valores |
|----------|------|-------------|---------|
| DESERCION | Binaria | Riesgo de deserción | 0 (sin riesgo), 1 (en riesgo) |

**Criterio de riesgo**: Un estudiante está en riesgo si tiene **PROMEDIO < 7.0 Y ASISTENCIA < 70%**

## 🎨 Características de la Interfaz

- **Diseño moderno** con esquema de colores profesional
- **Visualizaciones interactivas** con gráficos de Plotly
- **Navegación intuitiva** mediante sidebar
- **Cards informativas** con métricas destacadas
- **Responsive design** adaptable a diferentes pantallas
- **Estilo personalizado** con CSS


## 📝 Documentación Adicional

- **Código Comentado**: `desercion_app.py`
- **Dataset**: `REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv`

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## Link de la app en Streamlit

https://prediccion-desercion-estudiantil-o.streamlit.app/


