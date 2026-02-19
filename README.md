Complex Polar Transformer (Beta)
================================

Resumen
-------

Complex Polar Transformer es una implementación experimental de un modelo de
aprendizaje profundo para la predicción de propiedades moleculares sobre el
benchmark QM9. El modelo explora el uso de representaciones complejas en
coordenadas polares (magnitud y fase) junto con mecanismos de atención
compleja-polar, con el objetivo de capturar relaciones geométricas y angulares
entre átomos.

Este repositorio acompaña un trabajo de tesis de maestría y debe considerarse
código de investigación en fase beta.


Estado del proyecto
-------------------

- Estado: Beta / Experimental
- La arquitectura y la API interna pueden cambiar.
- El código prioriza claridad experimental y reproducibilidad sobre estabilidad
  de producción.


TL;DR (rápido)
--------------

1. Coloque los archivos del dataset QM9 en el directorio `data/`:
   - qm9.sdf
   - qm9_filtered_clean.csv
2. Ajuste la configuración del experimento en el archivo YAML:
   - experiments/beta_train.yaml
3. Ejecute el entrenamiento:
   - python main_train.py
4. Ejecute inferencia:
   - python predict.py --model checkpoints/model_epoch_X.pt


Requisitos
----------

- Python 3.9 – 3.10 (recomendado)
- PyTorch (CPU o CUDA)
- RDKit
- NumPy
- Pandas
- Matplotlib

Instalación mínima con pip:

    pip install torch numpy pandas matplotlib rdkit-pypi

Nota: para entrenamiento con GPU se recomienda instalar PyTorch con soporte CUDA.


Instalación recomendada (conda)
-------------------------------

    conda create -n complex-polar python=3.9 -y
    conda activate complex-polar
    pip install -r requirements.txt


Datos
-----

El proyecto utiliza el benchmark QM9:

- Archivo SDF: data/qm9.sdf
- Archivo CSV con propiedades: data/qm9_filtered_clean.csv

Durante la carga del dataset:
- Las moléculas no sanitizables por RDKit se descartan automáticamente.
- El número de moléculas válidas se reporta por consola.


Configuración del experimento
-----------------------------

La configuración se define mediante un archivo YAML. Ejemplo:

    dataset:
      sdf: "data/qm9.sdf"
      csv: "data/qm9_filtered_clean.csv"
      target: "u0"

    sample_size: 10000
    seed: 42

    batch_size: 16
    validation_split: 0.1
    learning_rate: 0.001
    max_epochs: 300

    model:
      in_dim: 5
      hidden_dim: 256
      out_dim: 1

Este enfoque permite:
- Reproducibilidad de los experimentos
- Trazabilidad de hiperparámetros
- Separación clara entre código y configuración


Entrenamiento
-------------

Para entrenar el modelo:

    python main_train.py

Durante el entrenamiento:
- Se guardan checkpoints en el directorio `checkpoints/`
- Se reportan métricas de validación (MAE, RMSE y R²)
- Se guarda la normalización del target para uso posterior en inferencia


Inferencia / Predicción
-----------------------

Ejemplo de uso:

    python predict.py \
      --sdf data/qm9.sdf \
      --csv data/qm9_filtered_clean.csv \
      --model checkpoints/model_epoch_300.pt \
      --output results/predictions.csv

El script de predicción:
- Desnormaliza las predicciones
- Calcula métricas de desempeño
- Genera una gráfica de Predicciones vs Valores reales


Estructura del repositorio
--------------------------

    .
    ├── core/           # dataset, trainer, métricas y utilidades
    ├── models/         # modelo complejo-polar y capas
    ├── experiments/    # archivos YAML de configuración
    ├── data/           # QM9 y archivos auxiliares
    ├── checkpoints/    # modelos entrenados (no versionados)
    ├── results/        # predicciones y gráficas
    ├── main_train.py
    ├── predict.py
    └── README.txt


Reproducibilidad
----------------

- Se fija una semilla global (seed) para todos los experimentos.
- La normalización del target se calcula únicamente con el conjunto de
  entrenamiento.
- Los hiperparámetros y estadísticas se almacenan en cada checkpoint.


Licencia y citación
-------------------

Incluya aquí la licencia correspondiente (por ejemplo MIT).

Si este código se utiliza en trabajos académicos, por favor cite el trabajo
asociado a la tesis.


Contacto
--------

Para reportar errores o realizar sugerencias:
- utilice el sistema de issues del repositorio
- o contacte directamente al autor del proyecto
