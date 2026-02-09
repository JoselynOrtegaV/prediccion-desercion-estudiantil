# 🎓 Sistema de Predicción de Deserción Estudiantil

## 📋 Descripción del Proyecto

Sistema de predicción de deserción estudiantil desarrollado con **Machine Learning** que utiliza la metodología **CRISP-DM** para identificar estudiantes en riesgo mediante el análisis de variables académicas clave.

El proyecto incluye:
- ✅ Modelo de clasificación basado en **K-Nearest Neighbors (KNN)**
- ✅ Técnica de balanceo **SMOTE** para mejorar la detección
- ✅ Interfaz web interactiva con **Streamlit**
- ✅ Visualizaciones con **Plotly** y análisis estadístico
- ✅ Documentación completa siguiendo **CRISP-DM**

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
7. Documentar el proceso metodológico siguiendo CRISP-DM

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
├── app.py                                    # Aplicación principal Streamlit
├── REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv  # Dataset (no incluido)
├── requirements.txt                          # Dependencias del proyecto
├── README.md                                 # Este archivo
│
└── .gitignore                               # Archivos ignorados por Git
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

## 🧮 Metodología CRISP-DM

El proyecto sigue las seis fases de CRISP-DM:

### 1. Comprensión del Negocio
- Identificación del problema de deserción estudiantil
- Definición de objetivos y criterios de éxito
- Énfasis en recall para minimizar falsos negativos

### 2. Comprensión de los Datos
- Análisis exploratorio del dataset
- Identificación de variables relevantes
- Evaluación de calidad de datos

### 3. Preparación de los Datos
- Limpieza de datos (valores nulos, formatos)
- Definición de variable objetivo: `DESERCION = 1 si (PROMEDIO < 7.0 Y ASISTENCIA < 70%)`
- Normalización con StandardScaler
- División train/test estratificada (70/30)

### 4. Modelado
- Algoritmo seleccionado: **K-Nearest Neighbors (KNN)**
- Parámetros: `n_neighbors=5`, `weights='distance'`
- Técnica de balanceo: **SMOTE** para manejar desbalance de clases

### 5. Evaluación
- Métricas: Accuracy, Precision, Recall, F1-Score
- Matriz de confusión
- Validación en conjunto de prueba

### 6. Despliegue
- Aplicación web interactiva con Streamlit
- Sistema de predicción en tiempo real
- Visualizaciones y recomendaciones

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

## 📊 Interpretación de Resultados

### Matriz de Confusión
- **Verdaderos Negativos (TN)**: Estudiantes sin riesgo correctamente identificados
- **Falsos Positivos (FP)**: Estudiantes sin riesgo marcados como en riesgo (falsa alarma)
- **Falsos Negativos (FN)**: ⚠️ **CRÍTICO** - Estudiantes en riesgo NO detectados
- **Verdaderos Positivos (TP)**: Estudiantes en riesgo correctamente detectados

### Por qué priorizamos el Recall
El **Recall alto (85-90%)** es fundamental porque:
- Minimiza estudiantes en riesgo no detectados (FN)
- Permite intervenciones tempranas
- Algunas falsas alarmas (FP) son preferibles a casos perdidos
- Facilita apoyo preventivo a más estudiantes



## 🌟 Mejoras Futuras

### Modelo
- [ ] Incorporar variables adicionales (socioeconómicas, participación)
- [ ] Probar algoritmos alternativos (Random Forest, XGBoost, Redes Neuronales)
- [ ] Implementar validación cruzada
- [ ] Optimización de hiperparámetros con Grid Search

### Interfaz
- [ ] Carga de archivos CSV personalizados
- [ ] Exportación de reportes en PDF
- [ ] Sistema de alertas automáticas
- [ ] Análisis de tendencias temporales

### Despliegue
- [ ] Contenedorización con Docker
- [ ] Despliegue en la nube (Heroku, AWS, Google Cloud)
- [ ] API REST para integración con otros sistemas
- [ ] Aplicación móvil


## 📝 Documentación Adicional

- **Código Comentado**: `app.py`
- **Dataset**: `REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.csv`

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 👥 Autores

Proyecto desarrollado como parte del curso de Minería de Datos.

