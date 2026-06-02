# Complex Polar Transformer — Benchmark QM9

Implementación experimental del modelo **ComplexPolarTransformer** para predicción de propiedades moleculares sobre QM9, con foco en la energía de atomización `u0_atom`.

Esta versión queda alineada con la configuración benchmark/v7 corregida:

- `RBFExpansion` usa distancias reales en Å.
- `cutoff` del YAML se aplica tanto al modelo como al dataset.
- `per_atom_norm` calcula estadísticas exactas sobre `u0_atom / N_atoms`.
- `predict_benchmark.py` reconstruye la arquitectura desde el checkpoint y evita cargas parciales por defecto.
- `predict.py` queda como wrapper legacy hacia `predict_benchmark.py`.

> Nota importante: los checkpoints entrenados antes de estas correcciones pueden cargar por forma de parámetros, pero no son equivalentes metodológicamente, porque antes el RBF recibía `dist/max_radius` y ahora recibe distancia real en Å. Para reportar métricas oficiales con esta versión corregida, reentrena el modelo.

---

## Estructura principal

```txt
core/
  dataset.py              # QM9SDFDataset corregido: distancia real en Å + cutoff real
  trainer_benchmark.py    # entrenamiento benchmark con per_atom_norm exacto
  collate.py              # collates reutilizables
models/
  complex_layers.py       # RBF/cosine cutoff corregidos
  complex_model_beta.py   # forward robusto para aristas vacías/None
experiments/
  beta_train_benchmark.yaml
main_train_benchmark.py
predict_benchmark.py
predict.py                # wrapper legacy
tests/
  test_forward.py
```

---

## Dataset esperado

Coloca los archivos en `data/`:

```txt
data/qm9.sdf
data/qm9.csv
```

El CSV debe incluir la columna:

```txt
u0_atom
```

Para esta versión, `u0_atom` se asume en **kcal/mol**, tal como está en tu CSV procesado.

---

## Entrenamiento oficial v7 corregido

Configura:

```txt
experiments/beta_train_benchmark.yaml
```

Ejecuta:

```bash
python main_train_benchmark.py
```

La configuración principal queda así:

```yaml
model:
  in_dim: 5
  hidden_dim: 256
  out_dim: 1
  num_hidden_layers: 3
  num_rbf: 150
  cutoff: 7.0
  edge_dim: 4
  dropout: 0.1
  use_residuals: true
  use_layernorm: true

normalize_target: true
per_atom_norm: true
grad_clip: 1.0
```

El entrenamiento guarda:

```txt
checkpoints/best_model.pt
logs/split_seed42.json
logs/training_log.csv
logs/loss_curve.png
```

---

## Predicción / evaluación benchmark

Comando oficial para `u0_atom`:

```bash
python predict_benchmark.py \
  --sdf data/qm9.sdf \
  --csv data/qm9.csv \
  --target u0_atom \
  --model checkpoints/best_model.pt \
  --split-file logs/split_seed42.json \
  --split test \
  --unit kcal \
  --batch-size 64 \
  --output results/predictions_v7_corrected.csv \
  --plot results/pred_vs_real_v7_corrected.png
```

El script reconstruye automáticamente:

- `num_hidden_layers`
- `num_rbf`
- `cutoff`
- `dropout`
- `use_residuals`
- `use_layernorm`
- `per_atom_norm`
- `y_mean` y `y_std`

No se permite carga parcial del checkpoint por defecto. Para depuración existe:

```bash
--allow-partial-load
```

No uses esa opción para reportar métricas oficiales.

---

## Prueba rápida del forward

```bash
python tests/test_forward.py
```

Debe responder:

```txt
test_forward: OK — output shape: torch.Size([2, 1])
```

---

## Cambios metodológicos aplicados

### 1. Cutoff real

Antes, `experiments/beta_train_benchmark.yaml` decía `cutoff: 7.0`, pero el dataset seguía usando `max_radius=5.0` por defecto. Ahora `main_train_benchmark.py` pasa `cutoff` al dataset:

```python
dataset = QM9SDFDataset(..., max_radius=cutoff)
```

### 2. Distancia real para RBF

Antes, `edge_attr[:, 0]` era:

```python
dist / self.max_radius
```

Ahora es:

```python
dist
```

Por tanto, las gaussianas RBF están centradas en Å reales dentro de `[0, cutoff]`.

### 3. Cosine cutoff corregido

Ahora la contribución fuera del cutoff se anula explícitamente:

```python
inside = (dist < self.cutoff).float()
cos_cutoff = 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0)
cos_cutoff = cos_cutoff * inside
```

### 4. Normalización per-atom exacta

Antes se aproximaba dividiendo estadísticas por `n_mean = 13.0`. Ahora se calcula directamente:

```txt
mean(u0_atom / N_atoms)
std(u0_atom / N_atoms)
```

usando únicamente el split de entrenamiento.

### 5. Predicción segura

`predict_benchmark.py` ahora carga primero el checkpoint, lee la configuración del modelo y luego crea el dataset con el mismo `cutoff` usado en entrenamiento.

---

## Reproducibilidad

El split se guarda en:

```txt
logs/split_seed42.json
```

Incluye:

- `n_total`
- `n_train`
- `n_val`
- `n_test`
- `target`
- `cutoff`
- `num_rbf`
- `per_atom_norm`
- índices de train/val/test

---

## Resultado anterior vs versión corregida

El resultado anterior `MAE ≈ 2.919 kcal/mol` corresponde al código previo. Después de esta corrección, debes reentrenar para obtener una métrica metodológicamente consistente con:

```txt
RBF×150 + cutoff real 7 Å + normalización per-atom exacta
```
