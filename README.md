# Predicción de enfermedad cardíaca — Prototipo académico

Aplicación en Streamlit del proyecto de Minería de Datos (tercer corte).

> ⚠️ **Prototipo académico.** No sustituye el diagnóstico ni la decisión de un profesional de la salud.

**App pública:** _(pegar aquí la URL de Streamlit Community Cloud)_

## Integrantes
Rafael sierra
## Descripción
Clasificación binaria de `heart_disease` (0 = no, 1 = sí) a partir de 15 variables clínicas y demográficas
de `patient_dataset.csv` (6.000 registros). **Fuente de los datos:** _(completar con el origen del dataset)_.

## Contenido del repositorio
| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Streamlit |
| `pipeline_modelo.joblib` | Pipeline completo (preprocesamiento + Random Forest ajustado) |
| `metrics.json` | Métricas finales y matriz de confusión |
| `muestra_prueba.csv` | 30 registros de prueba con la predicción del notebook |
| `patient_dataset.csv` | Datos para el EDA y los rangos del formulario |
| `analisis_patient_dataset.ipynb` | Notebook actualizado (secciones 13–15 nuevas) |
| `requirements.txt` | Dependencias (scikit-learn fijado a la versión de entrenamiento) |

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Revisión del modelo (tercer corte)
Todos los modelos del segundo corte tenían AUC ≈ 0,50, por debajo del umbral de 0,70, por lo que se aplicó
la ficha de revisión de calidad. El pipeline es correcto (partición estratificada, preprocesamiento dentro del
pipeline, validación cruzada solo sobre entrenamiento, sin fuga de información). El diagnóstico muestra que
**los datos no contienen señal predictiva**:

- Ninguna variable numérica supera |r| = 0,02 con el objetivo; la información mutua máxima es 0,017.
- La tasa de enfermedad por categoría está entre 48 % y 52 %.
- Prueba de permutaciones: AUC real 0,503 vs. 0,497 con etiquetas barajadas (p = 0,32).
- Las variables tienen límites redondos y distribuciones casi uniformes (compatible con datos sintéticos).

### Antes y después de la optimización (conjunto de prueba)
| Modelo | Accuracy | Recall (clase 1) | F1 | AUC-ROC |
|---|---|---|---|---|
| Random Forest (segundo corte) | 0,506 | 0,478 | 0,491 | 0,500 |
| **Random Forest (ajustado, modelo final)** | 0,504 | 0,472 | 0,487 | 0,506 |
| Gradient Boosting | 0,502 | 0,502 | 0,501 | 0,508 |
| Regresión logística (ajustada) | 0,489 | 0,441 | 0,463 | 0,476 |

Validación cruzada del modelo final (5 pliegues): accuracy 0,510 ± 0,023; recall 0,496 ± 0,042; AUC 0,518 ± 0,013.
Matriz de confusión (prueba): `[[323, 279], [316, 282]]`.

**Conclusión:** las diferencias entre modelos están dentro del ruido; el modelo no tiene capacidad predictiva útil.
Se eligió el Random Forest ajustado por tener el mayor AUC en validación cruzada y un recall estable de la clase positiva.

## Uso de inteligencia artificial
- **Herramienta utilizada:** Claude (Anthropic).
- **Propósito:** revisar el rendimiento del pipeline, diagnosticar la ausencia de señal en los datos, generar el código de optimización/exportación y la estructura de la app en Streamlit.
- **Prompt principal:** _(pegar aquí el prompt principal utilizado)_
- **Cómo se verificó la respuesta:** se ejecutó el notebook completo con el CSV; se comprobó que el pipeline guardado reproduce exactamente las predicciones del notebook en 30 registros de prueba; se probaron un caso normal y entradas inválidas (presión 400, edad negativa, categoría inexistente); se revisaron los resultados del diagnóstico.
- **Problemas encontrados:** el rendimiento de todos los modelos es cercano al azar; la versión de scikit-learn debe coincidir entre entrenamiento y despliegue; las categorías deben codificarse igual que en el entrenamiento (`"1.0"`, `"3"`, `"Desconocido"`). _(añadir los problemas que encuentren al desplegar)_
- **Ajustes realizados:** se fijó `scikit-learn` en `requirements.txt`, se replicó la codificación de categorías en la app y se documentó el resultado negativo en lugar de forzar la métrica. _(añadir otros ajustes)_

> Todos los integrantes deben comprender el código entregado.
