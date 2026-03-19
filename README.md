# Segmentación Semántica de Tumores Cerebrales mediante Arquitectura Híbrida U-Net y ResNet50

## Abstracto Científico
Este repositorio contiene la implementación de una red neuronal convolucional diseñada para la segmentación automatizada de neoplasias a partir de imágenes de resonancia magnética (MRI). El modelo utiliza una arquitectura híbrida que integra un codificador **ResNet50** preentrenado (Transfer Learning vía ImageNet) para la extracción robusta de características, acoplado a un decodificador simétrico tipo **U-Net** para la reconstrucción de la resolución espacial. 

El proyecto aborda el desafío inherente del desbalance de clases en la imagenología médica mediante la optimización estocástica de una función de pérdida combinada, logrando aislar la morfología patológica con alta precisión espacial.

---

## Estructura del Repositorio
* `notebooks/Trabajo_final_vision.ipynb`: Cuaderno de Jupyter con el pipeline completo (ETL, entrenamiento, inferencia y evaluación visual).
* `docs/`: Documentación técnica extendida y análisis de aplicaciones clínicas.
* `assets/`: Directorio destinado a las matrices de superposición cualitativa y gráficas de convergencia.
* `requirements.txt`: Dependencias estandarizadas del entorno de ejecución.

---

## Metodología y Arquitectura

### 1. Ingesta de Datos (Data Pipeline)
Los datos son ingeridos dinámicamente mediante la API de Kaggle, empleando el *dataset* `brain-tumor-image-dataset-semantic-segmentation`. El pipeline transforma las anotaciones vectoriales (formato COCO JSON) en máscaras binarias ráster y estructura conjuntos de datos optimizados utilizando `tf.data.Dataset` con operaciones de precarga (*prefetch*) y paralelización (*AUTOTUNE*).

### 2. Estrategia de Optimización y Entrenamiento
Para mitigar el desbalance espacial (donde el tejido sano representa >95% de los píxeles), el grafo computacional fue compilado empleando una **Pérdida Combinada (Combo Loss)**. Esta función integra la estabilidad termodinámica de la entropía cruzada con la penalización espacial estricta del Coeficiente Dice:

$$L_{combo} = L_{BCE} + \left(1 - \frac{2 \sum y_{true} y_{pred} + \epsilon}{\sum y_{true} + \sum y_{pred} + \epsilon}\right)$$

Se implementaron mecanismos heurísticos de regularización por retrollamada (*callbacks*):
* `ModelCheckpoint`: Serialización estricta del mínimo global empírico.
* `ReduceLROnPlateau`: Modulación dinámica del decaimiento de la tasa de aprendizaje.
* `EarlyStopping`: Prevención de sobreajuste (*overfitting*) basada en la divergencia de la pérdida de validación.

---

## Resultados y Auditoría Clínica

### Evaluación Cuantitativa
El rendimiento del modelo, tras alcanzar la convergencia óptima en la iteración 8 (previo a la activación de la parada temprana heurística en iteraciones posteriores), arroja las siguientes métricas globales:

| Métrica | Valor Obtenido | Relevancia Clínica |
| :--- | :--- | :--- |
| **Accuracy Global (Train)** | `0.9685` | Convergencia del modelo sobre la distribución de entrenamiento. |
| **Accuracy Global (Val)** | `0.9663` | Capacidad de generalización sobre datos no vistos. |
| **Loss (BCE)** | `0.1033` | Mínimo global empírico alcanzado en la topología de la función de pérdida. |

*(Nota técnica: El Accuracy superior al 96% refleja la correcta segmentación de la clase mayoritaria correspondiente al tejido de fondo. Para la evaluación de la superposición de la clase minoritaria (tejido tumoral), se requiere el análisis de la matriz de evaluación espacial multicanal (Overlay)).*

### Análisis de Robustez ante Ruido de Etiquetas (Noisy Labels)
El análisis cualitativo evidenció una alta resiliencia frente a anotaciones humanas imperfectas. Durante la inferencia, se detectó que el *Ground Truth* original presentaba deficiencias geométricas severas (uso de *bounding boxes* rectangulares que incluían tejido cerebral sano y fluido cerebroespinal patológico). 

En contraste, el campo receptivo de la arquitectura U-Net no colapsó hacia la memorización de esta geometría ruidosa. El modelo logró segmentar de manera orgánica la verdadera textura e intensidad radiológica de la masa sólida tumoral, excluyendo los falsos positivos anatómicos presentes en la etiqueta original. Este fenómeno valida la viabilidad clínica del extractor de características ResNet50 en escenarios con datos no depurados.

---

## Reproducibilidad del Experimento

1. Clonar el repositorio.
2. Instalar las dependencias estrictas: `pip install -r requirements.txt`.
3. Proveer el token de autenticación de Kaggle (`kaggle.json`) en el directorio raíz o configurar las variables de entorno correspondientes.
4. Ejecutar el cuaderno principal. El modelo generará dinámicamente una matriz estocástica multicanal de 4 dimensiones (Input, Target, Predicción IA y Overlay Clínico) para auditar la superposición anatómica.

**Documentación Extendida y Aplicaciones Clínicas:** [Consultar el Informe (PDF)](./docs/su_archivo_informe.pdf)
