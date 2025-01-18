# Análisis de sentimientos en reseñas de IMDB

Este proyecto utiliza técnicas de Procesamiento de Lenguaje Natural (NLP) y aprendizaje automático para clasificar el sentimiento de reseñas de películas como **positivas** o **negativas**.  

## Tabla de contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Dataset](#dataset)
- [Procesamiento de Datos](#procesamiento-de-datos)
- [Modelos Implementados](#modelos-implementados)
- [Evaluación de Modelos](#evaluación-de-modelos)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instrucciones de Ejecución](#instrucciones-de-ejecución)
- [Resultados](#resultados)
- [Mejoras Futuras](#mejoras-futuras)

---

## Descripción del proyecto

El objetivo principal del proyecto es construir un modelo que pueda clasificar reseñas de películas basándose en su contenido textual. Se trabaja con un dataset balanceado y desbalanceado, aplicando varias técnicas de vectorización y modelos de aprendizaje supervisado para la predicción de sentimientos.

---

## Estructura del proyecto

```plaintext
├── data/
│   ├── crudo/
│   │   └── IMDB_Dataset.csv  # Dataset original
├── notebooks/
│   └── nlp_sentiment_analysis.ipynb  # Código principal del análisis y modelos entrenados
├── src/
│   └── requirements.txt/  # Dependencias
├── README.md  # Este archivo
```

---

## Dataset

El dataset utilizado es el **IMDB Dataset of 50K Movie Reviews**, que contiene 50,000 reseñas etiquetadas como positivas o negativas.

### Distribución inicial:

- **25,000** reseñas positivas
- **25,000** reseñas negativas

### Transformaciones:
Se desbalanceó el dataset con un subconjunto de 9,000 reseñas positivas y 1,000 negativas.
Posteriormente, se balanceó utilizando técnicas de submuestreo

---

## Procesamiento de datos

1. **Limpieza del Dataset:** Se eliminan caracteres especiales y se procesa el texto para el análisis.
2. **Representación del Texto:**
- **CountVectorizer:** Representación basada en frecuencia de palabras.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Asigna pesos a palabras según su importancia.
3. **División del Dataset:**
- **67%** para entrenamiento.
- **33%** para prueba.

---

## Modelos implementados

Se implementaron los siguientes modelos de clasificación:

1. **Support Vector Machine (SVM)**
2. **Árbol de Decisión**
3. **Naive Bayes**
4. **Regresión Logística**

---

## Evaluación de modelos

Los modelos fueron evaluados utilizando las siguientes métricas:

- **Exactitud (Accuracy)**
- **F1 Score**
- **Reporte de Clasificación**
- **Matriz de Confusión**

## Resultados clave:

```markdown
| Modelo              | Exactitud  | F1 Score (Promedio) |
|---------------------|------------|---------------------|
| SVM                 | 83.6%      | 0.84                |
| Árbol de Decisión   | 68.4%      | 0.68                |
| Naive Bayes         | 65.0%      | 0.65                |
| Regresión Logística | 81.9%      | 0.82                |
```

---

## Requisitos del Sistema

- Python 3.8 o superior
- Librerías:
    - pandas
    - scikit-learn
    - imblearn

---

## Instrucciones de Ejecución

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/imdb_proyect.git
cd imdb_proyect
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Abre el notebook Jupyter:
```bash
jupyter notebook notebooks/nlp_sentiment_analysis.ipynb
```

---

## Resultados

El modelo **SVM con kernel lineal** mostró el mejor desempeño, logrando una precisión del 83.6% y un F1 Score promedio de 0.84. La matriz de confusión mostró un buen equilibrio entre las clases positivas y negativas.

---

## Mejoras Futuras

- Implementar técnicas avanzadas de NLP como modelos preentrenados (BERT o GPT).
- Incluir más datos para aumentar la robustez del modelo.
- Aplicar técnicas de optimización de hiperparámetros.