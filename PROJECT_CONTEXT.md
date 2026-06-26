El proyecto corresponde a una tesis de ciencias computacionales enfocada en el desarrollo de un modelo avanzado de Deep Learning para predicción de propiedades moleculares utilizando el benchmark QM9.

La propuesta se centra en diseñar un Transformer complejo con representación polar capaz de modelar información molecular tridimensional utilizando:

- números complejos
- geometría molecular
- coordenadas polares/esféricas
- relaciones radiales y angulares
- mecanismos de atención compleja

El objetivo principal es predecir propiedades fisicoquímicas y fisicomecánicas de moléculas del dataset QM9 utilizando una arquitectura más eficiente y físicamente interpretable que modelos E(3)-equivariantes tradicionales.

La hipótesis del proyecto plantea que:

“Un Transformer formulado en el dominio complejo y complementado con una representación polar que capture explícitamente magnitudes, fases y dependencias radiales-angulares puede mejorar la precisión, interpretabilidad física y capacidad de generalización frente a modelos tradicionales y modelos E(3)-equivariantes.”

El modelo debe:

- usar números complejos reales y coherentes
- preservar magnitud y fase
- preservar orientación molecular
- modelar explícitamente relaciones radiales y angulares
- integrar coordenadas polares/esféricas
- usar atención compleja-polar
- mantener coherencia física
- mantener coherencia algebraica compleja
- preservar información geométrica tridimensional
- mantener invariancia traslacional y rotacional
- mejorar eficiencia computacional frente a modelos como:
  - NequIP
  - Allegro
  - TensorNet
  - MGNN
  - PaiNN
  - ViSNet

El proyecto NO busca convertirse en:

- un GNN convencional
- un Transformer estándar real
- un modelo cartesiano clásico
- un SE(3)-Transformer pesado
- un modelo complejo decorativo

La arquitectura debe seguir siendo:

- compleja
- polar
- geométrica
- físicamente interpretable

La fase NO puede perder significado físico.

La idea científica central del proyecto es:

“Codificar orientación molecular mediante fase compleja físicamente interpretable.”

Usando representaciones del tipo:

z = r·e^(iθ)

donde:

- la magnitud representa intensidad/interacción física
- la fase representa orientación y relaciones geométricas moleculares

Actualmente el modelo ya implementa:

- embeddings complejos
- representación polar parcial
- operaciones complejas
- atención compleja
- información geométrica molecular

Sin embargo, actualmente presenta limitaciones importantes:

- MAE aproximado ≈ 0.9
- pérdida de información angular
- pooling molecular débil
- representación química incompleta
- posible colapso de fase compleja
- geometría molecular insuficiente
- falta de interacciones de orden superior
- posible destrucción de información polar
- cuellos de botella computacionales

El proyecto utiliza el dataset QM9, que contiene aproximadamente 134 mil moléculas orgánicas con coordenadas 3D y propiedades cuánticas calculadas mediante DFT.

Las propiedades objetivo incluyen:

- energía interna (U0)
- HOMO/LUMO
- gap energético
- dipole moment
- polarizabilidad
- propiedades termodinámicas

El objetivo técnico actual es reducir el MAE desde aproximadamente 0.9 hacia:

- 0.7
- 0.5
- idealmente < 0.4

sin romper:

- la hipótesis científica
- el dominio complejo
- la representación polar
- la interpretación física de la fase
- la coherencia matemática

El proyecto está desarrollado utilizando:

- Python
- PyTorch
- PyTorch Geometric
- CUDA
- RDKit
- ASE
- NumPy
- SciPy

El entrenamiento principal se realiza mediante:

python main_train_benchmark.py

La inferencia se realiza mediante:

python predict_benchmark.py \
  --sdf data/qm9.sdf \
  --csv data/qm9.csv \
  --target u0_atom \
  --model checkpoints_v9_angular_modrelu/best_model.pt \
  --split-file logs_v9_angular_modrelu/split_seed42.json \
  --split test \
  --unit kcal \
  --output results/predictions_v9_angular_modrelu.csv \
  --plot results/pred_vs_real_v9_angular_modrelu.png

El anteproyecto define evaluación mediante:

- MAE
- RMSE
- R²

además de métricas computacionales como:

- throughput
- latencia
- tiempo por época
- memoria GPU
- número de parámetros
- eficiencia computacional

La tesis también plantea:

- comparación contra NequIP, Allegro, TensorNet, MGNN y ViSNet
- estudios de ablación
- evaluación Out-of-Distribution (OOD)
- interpretabilidad física de la fase compleja
- análisis de eficiencia computacional

Actualmente el proyecto requiere una refactorización profunda y una mejora matemática, geométrica y arquitectónica significativa para alcanzar resultados competitivos y científicamente sólidos.

La siguiente etapa del proyecto consiste en diseñar una nueva generación del modelo:

Complex Polar Geometric Transformer V2/V3

con:

- atención polar geométrica
- fase físicamente interpretable
- message passing geométrico complejo
- relaciones angulares avanzadas
- pooling físico
- mejor estabilidad compleja
- mayor eficiencia computacional
- menor MAE
- mejor generalización
- mejor preservación geométrica molecular