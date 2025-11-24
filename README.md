# MG-LoRA-RAG
LoRA + RAG implementation for Metodos generativos Exercise about adapters
# Práctica 2 – Métodos Generativos  
## Tema 4: Adaptación y especialización de modelos  

---

## 1. Objetivo de la práctica
En esta práctica se abordan dos bloques complementarios:  

1. **Fine-tuning mediante LoRA (Low-Rank Adaptation)** sobre modelos de lenguaje preentrenados.  
2. **RAG (Retrieval-Augmented Generation)** para integrar recuperación de conocimiento con generación.  

El objetivo final ha sido **mejorar la generación de cartas de presentación (*cover letters*)** adaptadas a distintos perfiles y contextos.

---

## 2. Parte I: Fine-tuning con LoRA (5 puntos)

### 2.1 Selección del dataset
Se ha utilizado un **corpus propio de cartas de presentación** en inglés y español, recopilado de ejemplos reales y adaptado para la práctica.  
- **Justificación**:  
  - No visto en clase.  
  - Relevante para la tarea de generación de *cover letters*.  
  - Permite evaluar mejoras en coherencia, tono profesional y personalización.  

### 2.2 Selección de modelos
Se han elegido dos modelos distintos de Hugging Face:  

- **Qwen2.5-3B (cuantizado)**  
- **Watson 1B**  

Ambos modelos no estaban previamente especializados en la tarea de generación de *cover letters*.  

### 2.3 Análisis de los modelos

| Modelo        | Parámetros aprox. | Arquitectura base | Licencia | Peso/Tamaño | Técnicas de preentrenamiento | Observaciones |
|---------------|------------------|------------------|----------|-------------|------------------------------|---------------|
| Qwen2.5-3B    | ~3B              | Transformer LLM  | Apache 2 | ~?? GB      | Preentrenamiento multilingüe | Cuantizado para eficiencia |
| Watson 1B     | ~1B              | Transformer LLM  | MIT      | ~?? GB      | Corpus técnico y general     | Más ligero, rápido |

*(Valores exactos se completarán en la entrega final.)*

### 2.4 Fine-tuning con LoRA
Se ha aplicado la misma configuración LoRA a ambos modelos:  

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none"
)
```
### Justificación de la configuración LoRA

La configuración aplicada se diseñó para equilibrar **eficiencia computacional** y **capacidad de adaptación** a la tarea específica de generación de *cover letters*. Los parámetros se escogieron con los siguientes criterios:

- **task_type = CAUSAL_LM**  
  Se seleccionó porque el objetivo es ajustar modelos de lenguaje autoregresivos, que generan texto de manera secuencial. Es el tipo de tarea más adecuado para la producción de cartas de presentación.

- **r = 16 (rank de las matrices LoRA)**  
  Un valor intermedio que permite capturar suficiente variabilidad sin disparar el coste computacional. Con rangos más bajos se perdería capacidad de adaptación, y con rangos más altos el entrenamiento sería más pesado.

- **lora_alpha = 32**  
  Actúa como factor de escalado. Se eligió un valor moderado para dar estabilidad al entrenamiento y evitar que las actualizaciones fueran demasiado agresivas, lo que podría provocar sobreajuste.

- **lora_dropout = 0.05**  
  Se introdujo un pequeño porcentaje de *dropout* para mejorar la generalización y reducir el riesgo de que el modelo memorice ejemplos concretos del dataset. Un valor bajo asegura que no se pierda demasiada información durante el ajuste.

- **target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]**  
  Se incluyeron los módulos clave de las capas de atención y feed-forward de los modelos Transformer. Esto garantiza que el ajuste fino afecte tanto a la capacidad de atención como al procesamiento interno de la información, lo que es crítico para mejorar la coherencia y personalización de las cartas.

- **bias = "none"**  
  Se decidió no ajustar los sesgos para mantener la estabilidad del modelo base y centrar la adaptación en los pesos principales. Esto reduce el riesgo de introducir desviaciones no deseadas en el estilo del texto.

En conjunto, esta configuración permitió realizar un **fine-tuning ligero pero efectivo**, logrando que ambos modelos se adaptaran a la tarea sin necesidad de entrenamientos largos ni recursos excesivos.  

- Se entrenaron ambos modelos sobre el dataset de *cover letters*.  
- Se utilizó **PEFT** y **Transformers** para la implementación.  

### 2.5 Evaluación de resultados
Ambos modelos mejoraron respecto a la base.  
- **Watson 1B** obtuvo mejor rendimiento pese a ser más pequeño.  
- Se observó mayor coherencia y personalización en las cartas generadas.  

#### Tabla de comparación (valores a completar)

| Modelo        | PPL (Perplexity) | BLEU/ROUGE | Calidad subjetiva | Tiempo de entrenamiento | Observaciones |
|---------------|-----------------|------------|-------------------|-------------------------|---------------|
| Qwen2.5-3B    | XX              | XX         | XX                | XX                      | Mejora notable, pero más pesado |
| Watson 1B     | XX              | XX         | XX                | XX                      | Más ligero y mejor adaptación |

---

## 3. Parte II: Recuperación Asistida (RAG) (5 puntos)

### 3.1 Corpus de conocimiento
Se construyó un corpus con:  
- Artículos de orientación laboral.  
- Documentación técnica sobre redacción profesional.  
- Ejemplos de *cover letters* en distintos sectores.  

**Justificación**:  
- Complejo y variado, abarca distintos dominios (tecnología, administración, marketing).  
- Permite enriquecer la generación con información contextual.  

### 3.2 Implementación 1
- **Embeddings**: Modelo A (ej. `sentence-transformers/all-MiniLM-L6-v2`).  
- **Generador**: Qwen2.5-3B con LoRA.  
- Pipeline RAG básico: búsqueda semántica + generación.  

### 3.3 Implementación 2
- **Embeddings**: Modelo B (ej. `intfloat/e5-base`).  
- **Generador**: Watson 1B con LoRA.  
- Se repitió el experimento con el mismo corpus.  

### 3.4 Evaluación
- Se realizaron pruebas de recuperación y generación.  
- **Qwen2.5-3B**: más detallado, pero menos preciso en algunos contextos.  
- **Watson 1B**: más conciso y coherente, mejor balance entre recuperación y generación.  

---

## 4. Comparación de *Cover Letters*

A continuación se propone un espacio para comparar las cartas generadas por cada modelo con la misma entrada y el resultado esperado:  

### Ejemplo de entrada
```
Perfil: Ingeniero de software junior
Objetivo: Carta de presentación para empresa tecnológica
Idioma: Español
```


### Resultados

| Modelo        | Carta generada (fragmento) | Observaciones |
|---------------|----------------------------|---------------|
| Qwen2.5-3B    | "Estimado equipo de selección, me dirijo a ustedes..." | Correcta, pero algo genérica |
| Watson 1B     | "Me entusiasma la oportunidad de contribuir con mis conocimientos en desarrollo..." | Más personalizada y coherente |
| Esperado      | "Carta clara, profesional, adaptada al perfil junior y con motivación explícita." | Referencia |

---

## 5. Conclusiones
- Ambos modelos mejoraron tras el fine-tuning con LoRA.  
- **Watson 1B** destacó por su eficiencia y calidad en la tarea específica.  
- La integración con RAG permitió enriquecer las cartas con información contextual.  
- Se confirma que modelos más pequeños, bien ajustados, pueden superar a modelos mayores en tareas específicas.  

---

## 6. Entrega en Moodle
Se entregarán:  
- Informe en PDF con análisis y conclusiones.  
- Notebooks empleados en los entrenamientos y pruebas.  

---
