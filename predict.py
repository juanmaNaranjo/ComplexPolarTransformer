"""
Script legacy conservado por compatibilidad.

Para resultados oficiales de la tesis/benchmark usa predict_benchmark.py.
Este wrapper redirige al flujo corregido para evitar reconstruir modelos v6/v7 sin
num_rbf/cutoff/per_atom_norm.
"""

from predict_benchmark import parse_args, predict


if __name__ == "__main__":
    print("[INFO] predict.py es legacy. Redirigiendo a predict_benchmark.py.")
    predict(parse_args())
