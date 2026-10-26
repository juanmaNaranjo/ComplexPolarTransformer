# ComplexPolarTransformer v11 QKV Attention

Esta versión parte de la v9 angular explícito + modReLU y agrega un mecanismo de atención tipo Transformer con proyecciones Q/K/V complejas aprendibles.

## Objetivo

Probar si formalizar la atención como Q/K/V mejora el rendimiento frente a la v9 oficial, sin modificar ni sobrescribir la v9.

## Cambio principal

La v9 usa `ComplexPolarAttention`, basada en similitud directa de magnitud y fase.

La v11 usa opcionalmente `ComplexPolarQKVAttention` mediante:

```yaml
model:
  attention_type: "qkv"
```

Internamente se aprende:

```text
Q = Wq(z)
K = Wk(z)
V = Wv(z)
score_ij = Re(<Q_i, K_j>) / sqrt(d)
```

Se mantiene el sesgo de aristas basado en RBF y el Message Passing angular explícito j-i-k.

## Entrenar v11 oficial

```bash
python main_train_benchmark.py --config experiments/beta_train_v11_qkv_attention.yaml
```

## Evaluar

```powershell
python predict_benchmark.py `
  --sdf data/qm9.sdf `
  --csv data/qm9.csv `
  --target u0_atom `
  --model checkpoints_v11_qkv_attention/best_model.pt `
  --split-file logs_v11_qkv_attention/split_seed42.json `
  --split test `
  --unit kcal `
  --output results/predictions_v11_qkv_attention.csv `
  --plot results/pred_vs_real_v11_qkv_attention.png
```

## Prueba rápida opcional

```bash
python main_train_benchmark.py --config experiments/beta_train_v11_qkv_attention_fast.yaml
```

La prueba rápida solo valida estabilidad; no debe reportarse como resultado oficial.
