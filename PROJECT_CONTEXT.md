# Complex Polar Geometric Transformer — Fundamentos Científicos del Proyecto

## 1. Enunciado del Problema Científico

El aprendizaje automático aplicado a química cuántica enfrenta un desafío estructural profundo: las propiedades moleculares dependen de la geometría tridimensional de la molécula, pero la información geométrica cruda —coordenadas cartesianas absolutas— viola las invariancias fundamentales del sistema físico.

Una molécula de etanol y la misma molécula trasladada un metro tienen exactamente las mismas propiedades energéticas. Una molécula de benceno y la misma molécula rotada 30 grados tienen exactamente el mismo gap HOMO-LUMO. Sin embargo, un modelo que consuma coordenadas absolutas las trataría como moléculas completamente distintas, obligando a la red a aprender que la energía es invariante a la traslación y a la rotación desde los datos. Esto despilfarra capacidad de representación en aprender trivialidades geométricas que la física garantiza a priori.

La solución obvia —usar descriptores invariantes— enfrenta a su vez un problema de expresividad: los descriptores puramente radiales (solo distancias) no distinguen isómeros estructurales, no capturan la direccionalidad química de los orbitales, y no modelan las interacciones angulares que determinan propiedades como el dipolo molecular, la polarizabilidad o la actividad biológica.

Este proyecto propone resolver esa tensión mediante una **representación polar relacional compleja**: una arquitectura donde la información geométrica molecular se codifica en el dominio de los números complejos, separando magnitud (intensidad de interacción) de fase (orientación relacional), y donde toda esa geometría se extrae de relaciones entre átomos —no de coordenadas absolutas—, garantizando invariancia a SE(3) sin recurrir al pesado aparato matemático de los modelos equivariantes.

---

## 2. Contexto Científico y Motivación

### 2.1 El Benchmark QM9 y la Escala del Problema

El dataset QM9 contiene ~134.000 moléculas orgánicas pequeñas (C, H, N, O, F) con sus coordenadas 3D calculadas mediante Teoría del Funcional de la Densidad (DFT) y sus propiedades cuánticas asociadas: energía interna U₀, gap HOMO-LUMO, dipolo eléctrico, polarizabilidad y propiedades termodinámicas.

El cálculo DFT de una sola molécula toma minutos u horas. Los métodos de aprendizaje automático prometen reducir ese tiempo a milisegundos, pero solo si son capaces de aproximar fielmente la hipersuperficie de energía potencial (PES) del sistema. Esta hipersuperficie es inherentemente geométrica: depende solo de las posiciones nucleares relativas (aproximación de Born-Oppenheimer).

El estado del arte en el benchmark QM9 para la propiedad U₀ se sitúa entre 0.22 y 0.35 kcal/mol para modelos que explotan geometría 3D completa.

### 2.2 Taxonomía de los Modelos Existentes

Los modelos de Machine Learning para química cuántica se dividen en tres generaciones según su capacidad de capturar invariancias y su expresividad geométrica:

**Primera generación — Fingerprints y descriptores globales:** Morgan fingerprints, Coulomb matrices, SOAP. Invariantes pero con pérdida severa de información geométrica. No capturan coordenadas 3D.

**Segunda generación — GNNs radiales:** SchNet, MPNN, DimeNet. Codifican distancias interatómicas con funciones de base radial (RBF). SchNet alcanza 0.313 kcal/mol pero usa solo información radial, perdiendo el efecto de los ángulos de enlace. DimeNet agrega ángulos de tripleta pero en representación real-valued, sin capacidad de representar interferencia geométrica.

**Tercera generación — Modelos equivariantes:** NequIP, Allegro, TensorNet, PaiNN, ViSNet. Implementan equivariancia completa al grupo SE(3) usando representaciones tensoriales irreducibles (harmónicos esféricos, coeficientes de Clebsch-Gordan). Altamente expresivos pero costosos computacionalmente (O(L³) en el grado máximo de momento angular L) y con interpretabilidad limitada.

### 2.3 El Espacio Vacío

Existe un espacio científico sin explorar entre la segunda y tercera generación: modelos que capturen la geometría angular de forma invariante, más eficientes que los equivariantes completos, y con representaciones físicamente interpretables. Este proyecto ocupa ese espacio mediante el dominio complejo.

---

## 3. La Representación Polar Relacional — Núcleo del Aporte Científico

### 3.1 Por qué NO basta con transformar a coordenadas polares

Una confusión frecuente en este campo es identificar "representación polar" con "transformación de coordenadas cartesianas a esféricas". Esta confusión es científicamente errónea y computacionalmente ineficiente.

Las coordenadas esféricas (r, θ, φ) calculadas desde el centroide molecular son **no invariantes a la rotación**: si se rota la molécula, θ y φ cambian, mientras que r solo cambia parcialmente. Un modelo que consuma (r, θ, φ) respecto al centroide está usando un sistema de referencia absoluto que viola la invariancia rotacional.

La representación polar que propone este proyecto es fundamentalmente distinta: es **relacional**. Los descriptores geométricos se construyen desde las relaciones entre pares y tripletas de átomos, no desde posiciones respecto a un origen externo.

### 3.2 La Geometría Molecular como Sistema de Coordenadas Internas

El álgebra de la mecánica molecular demuestra que la energía potencial de una molécula de N átomos depende de 3N−6 coordenadas internas (para moléculas no lineales): distancias de enlace, ángulos de enlace y ángulos diedros. Estas coordenadas son intrínsecamente invariantes a SE(3).

Este proyecto extrae información geométrica de dos tipos de relaciones invariantes:

**Distancias interatómicas** d_{ij} = ‖r_i − r_j‖₂: invariantes a traslación y rotación. Codificadas mediante funciones de base radial gaussiana con cutoff coseno:

```
rbf_k(d_{ij}) = exp(−γ(d_{ij} − μ_k)²) · ½(cos(πd_{ij}/c) + 1)
```

que garantizan suavidad en el cutoff y desvanecimiento correcto de las interacciones a larga distancia.

**Cosenos de ángulos de enlace** cos(θ_{jik}) = (r_{ij} · r_{ik}) / (‖r_{ij}‖ · ‖r_{ik}‖): donde r_{ij} = r_j − r_i es el vector de enlace desde el átomo i hacia el átomo j. Estos ángulos son invariantes a SE(3) porque el producto punto de vectores de enlace normalizados no cambia bajo rotaciones o traslaciones. Se codifican mediante funciones de base gaussiana angular:

```
abf_k(cos θ_{jik}) = exp(−γ(arccos(cos θ_{jik}) − μ_k)²)
```

Esta elección garantiza que la red nunca vea coordenadas absolutas —ni cartesianas ni polares globales—, sino únicamente las relaciones geométricas locales que la física identifica como las variables relevantes.

### 3.3 Superioridad de la Representación Relacional

La representación relacional es superior a las coordenadas absolutas en tres dimensiones:

**Eficiencia de representación**: un modelo que consuma coordenadas absolutas debe aprender la invariancia a SE(3) desde los datos, usando parámetros que de otro modo podrían dedicarse a aprender la física química. Un modelo con entradas intrínsecamente invariantes delega la geometría a los datos y usa sus parámetros exclusivamente para la física.

**Generalización fuera de distribución**: moléculas en orientaciones no vistas en el entrenamiento son un caso de distribución desplazada para modelos con coordenadas absolutas. Para el modelo relacional, son exactamente iguales a las vistas en entrenamiento.

**Interpretabilidad**: los descriptores relacionales (distancias, ángulos) corresponden directamente a cantidades observables en espectroscopía IR, NMR y difracción de rayos X. Los modelos que aprenden sobre estas cantidades producen representaciones más directamente interpretables.

---

## 4. El Dominio Complejo — Fundamento Matemático y Físico

### 4.1 Fundamento Matemático

Un número complejo z = r·e^{iθ} ∈ ℂ es una representación polar que separa exactamente dos tipos de información ortogonales:

- **r = |z| ≥ 0**: la magnitud, que captura intensidad, norma o "cuánto"
- **θ = arg(z) ∈ (−π, π]**: la fase, que captura orientación, dirección o "en qué sentido"

La estructura algebraica del dominio complejo ofrece operaciones con significado geométrico directo:

**Producto complejo:** z₁ · z₂ = r₁r₂ · e^{i(θ₁+θ₂)}

Las magnitudes se multiplican y las fases se suman. Esto significa que el producto complejo implementa naturalmente la composición de orientaciones: si z₁ representa la orientación de un enlace y z₂ la de otro, su producto representa la "orientación compuesta" y la diferencia de orientación queda codificada en θ₁ + θ₂.

**Conjugado:** z* = r·e^{−iθ}

El producto z_i · z_j* = r_i r_j · e^{i(θ_i − θ_j)} mide simultáneamente la similitud de magnitudes (a través de r_i r_j) y la diferencia de orientaciones (a través de θ_i − θ_j). Su parte real Re(z_i · z_j*) = r_i r_j cos(θ_i − θ_j) es el producto interno hermítico, que es máximo cuando las orientaciones coinciden (θ_i = θ_j) y nulo cuando son ortogonales.

**Exponencial compleja:** e^{iθ} parametriza el grupo de Lie SO(2) = U(1), el grupo de rotaciones en el plano. Una rotación de ángulo α en ℝ² corresponde a una multiplicación por e^{iα} en ℂ. Esto hace del dominio complejo el álgebra natural para representar rotaciones planares y, por extensión, orientaciones relativas.

### 4.2 Fundamento Físico

La elección del dominio complejo no es arbitraria: está justificada por la física subyacente al problema.

**Mecánica cuántica**: las funciones de onda moleculares Ψ(r) son inherentemente complejas. La aproximación DFT, en la que se calculan los datos de QM9, usa orbitales de Kohn-Sham que son funciones complejas en general (solo se simplifican a reales bajo ciertas simetrías). La energía total incluye términos de energía cinética que involucran el laplaciano de la función de onda compleja.

**Orbitales moleculares**: los orbitales atómicos (s, p, d) tienen estructura nodal que incluye cambios de fase. Un orbital p_z tiene fase positiva en z > 0 y negativa en z < 0. La formación de un enlace σ requiere superposición en fase; la formación de un enlace π requiere alineación de fases. Un modelo que represente átomos como números complejos puede naturalmente distinguir entre interacciones en fase (constructivas, enlazantes) y en contrafase (destructivas, antienlazantes).

**Polarización electrónica**: el momento dipolar de una molécula es un vector, que tiene magnitud (qué tan polar es) y dirección (en qué sentido está polarizada). La representación compleja z = r·e^{iθ} captura naturalmente esta dualidad magnitud-dirección.

**Interferencia**: en un sistema de espín cuántico o en una red de osciladores acoplados, la suma de amplitudes complejas captura interferencia constructiva y destructiva. En el contexto molecular, dos átomos con fases opuestas que contribuyen a la misma propiedad se cancelan (como dos contribuciones dipolares opuestas en una molécula simétrica). Esta cancelación es físicamente correcta y emerge naturalmente del dominio complejo sin necesitar codificación explícita.

### 4.3 La Fase como Variable Geométrica, No como Embedding Aprendido

La distinción central del aporte de este proyecto es que **la fase no es simplemente un parámetro libre aprendido**. Es una variable que se inicializa con información geométrica y evoluciona a través de capas de message passing con significado físico preservado:

**Inicialización**: la fase se inicializa desde las features atómicas (tipo de átomo, hibridación, aromaticidad, carga) mediante una red MLP que mapea química a orientación inicial: θ₀ = MLP_phase(features) · π ∈ (−π, π].

**Evolución en message passing**: durante el paso de mensajes, la fase evoluciona por modulación angular:

```
msg_phase_{ij} = f_rbf(d_{ij}) · θ_source + θ_angular_{ij}
```

donde θ_angular_{ij} = λ · π · g_angle(abf(cos θ_{jik})) codifica la geometría local del ángulo de enlace en la fase del mensaje. La fase acumula información angular a través de las capas, evolucionando desde orientaciones atómicas iniciales hacia orientaciones moleculares emergentes.

**Significado tras el message passing**: después de L capas de message passing, la fase de cada átomo codifica un resumen de su entorno geométrico local —los ángulos formados por sus vecinos, la orientación relativa de sus enlaces— calculado de forma inductiva y relacional. Esta fase no describe la orientación absoluta del átomo en el espacio (lo cual no sería invariante), sino la orientación relativa dentro de su entorno de vecindad.

---

## 5. Invariancias y Coherencia Física

### 5.1 Invariancia Traslacional

Se garantiza en el dataset mediante la centración en el centroide geométrico antes de calcular cualquier descriptor:

```
r_i ← r_i − (1/N) Σⱼ rⱼ
```

Todos los descriptores subsiguientes (distancias, ángulos, vectores de enlace) son calculados sobre estas coordenadas centradas, eliminando completamente la dependencia traslacional.

### 5.2 Invariancia Rotacional

Se garantiza mediante la elección de descriptores intrínsecos:

- **Distancias d_{ij}**: invariantes a SO(3) trivialmente, pues ‖R·r_i − R·r_j‖ = ‖r_i − r_j‖ para toda R ∈ SO(3).
- **Cosenos de ángulos cos θ_{jik}**: invariantes a SO(3) porque el producto punto de vectores se preserva bajo rotaciones: (R·r_{ij}) · (R·r_{ik}) = r_{ij}ᵀRᵀR·r_{ik} = r_{ij}·r_{ik}.
- **Distancia al centroide r_i = ‖r_i − centroid‖**: invariante a rotaciones (las normas no cambian bajo rotaciones).

Crucialmente, las coordenadas esféricas (θ, φ) calculadas respecto al centroide —que SÍ cambian con la rotación— están **explícitamente excluidas** del embedding atómico. Solo se usa r (invariante). Toda la información angular entra únicamente a través de los cosenos de ángulos de enlace en `ComplexMessagePassing`, que son invariantes por construcción.

### 5.3 Sin Equivariancia — Invariancia Suficiente

Los modelos equivariantes (NequIP, Allegro) garantizan que las representaciones intermedias se transformen de forma predecible bajo rotaciones del sistema: F(R·G) = ρ(R)·F(G). Esto es computacionalmente costoso (requiere contracciones de tensores con coeficientes de Clebsch-Gordan) pero permite que las representaciones intermedias sean "compatibles" con la geometría del espacio.

Este proyecto elige **invariancia** en lugar de equivariancia, lo que es teóricamente suficiente para predicciones de propiedades escalares (energía, HOMO-LUMO, polarizabilidad). Si la propiedad objetivo p(mol) es invariante a SE(3), y la representación z(mol) es invariante a SE(3), entonces una función f(z(mol)) puede aproximar p(mol) sin necesitar equivariancia en las capas intermedias.

Esta elección reduce dramáticamente la complejidad computacional: en lugar de contracciones de Clebsch-Gordan con escala O(L³) en el grado máximo de momento angular, se opera directamente con tensores complejos de dimensión fija H, con escala O(H²) o O(N²H) para la atención.

La coherencia física se preserva porque:
1. Los descriptores de entrada son invariantes a SE(3)
2. Las operaciones del modelo (atención, message passing, pooling) son simétricas respecto a la permutación de átomos
3. La fase evoluciona por modulación aditiva y multiplicativa de descriptores invariantes, por lo que nunca representa información no-invariante

---

## 6. El Transformer Complejo — Integración Arquitectónica

### 6.1 Atención Compleja Hermítica

En un Transformer estándar, el score de atención entre la query del átomo i y la key del átomo j es:

```
s(i, j) = (Q_i · K_j) / √d_k   ∈ ℝ
```

En el Transformer complejo polar, la query y la key son números complejos, y el score es el producto interno hermítico en ℂᴴ:

```
s(i, j) = Re(Q_i · conj(K_j)) / √(H/h) = Σₖ [q_real(i,k)·k_real(j,k) + q_imag(i,k)·k_imag(j,k)] / √(H/h)
```

Expandiendo:

```
s(i, j) = Σₖ |Q_{i,k}| · |K_{j,k}| · cos(∠Q_{i,k} − ∠K_{j,k}) / √(H/h)
```

Este score tiene una interpretación física directa: mide la **concordancia de magnitudes y fases** entre los átomos i y j. Dos átomos con magnitudes similares y fases alineadas (cos ≈ 1) tienen alto score; dos átomos con fases opuestas (cos ≈ −1) tienen score negativo.

Desde el punto de vista químico: átomos con características electrónicas similares (mismo tipo de orbital, misma orientación geométrica) se atienden mutuamente con mayor peso. Esto es análogo al criterio de similaridad en teoría de perturbaciones de la mecánica cuántica.

La implementación eficiente evita materializar los tensores intermedios [N, N, h, H/h]:

```python
# Representación cartesiana de Q y K
q_real = q_mag * cos(q_phase)   # [N, h, H/h]
q_imag = q_mag * sin(q_phase)

# Score mediante einsum (sin crear [N,N,h,H/h])
scores = (einsum("ihd,jhd->ijh", q_real, k_real) +
          einsum("ihd,jhd->ijh", q_imag, k_imag)) / scale   # [N, N, h]
```

### 6.2 Edge-Masking Geométrico

La atención se restringe al grafo molecular: solo los pares (i, j) dentro del cutoff d_{ij} ≤ c tienen score finito; los pares fuera del cutoff reciben score = −∞, que después de softmax produce peso = 0. Esto garantiza que la atención sea local (dentro del entorno de vecindad) y coherente con la física de interacciones de corto alcance.

El bias de arista rbf → h (un bias distinto por cabeza) modula el score de atención con información de la distancia, integrando la geometría radial directamente en el mecanismo de atención.

### 6.3 Message Passing Complejo con Modulación Angular

El paso de mensajes opera en forma polar y en espacio cartesiano de forma complementaria:

**Construcción del mensaje** (espacio polar):
```
msg_ij = (f_rbf(rbf_{ij}) * mag_j) · exp(i · (f_rbf_phase(rbf_{ij}) · π + phase_j))
```

La magnitud del mensaje escala el mensaje del átomo fuente j por la ponderación radial. La fase del mensaje rota la fase del átomo fuente j por un ángulo codificado en la distancia.

**Modulación angular** (características invariantes):
```
msg_mag_{ij}   *= (1 + λ_mag   · g_mag(abf(cos θ_{jik})))
msg_phase_{ij} += λ_phase · π · g_phase(abf(cos θ_{jik}))
```

Los ángulos cos(θ_{jik}) —calculados UNA vez por molécula, compartidos entre todas las capas— modulan tanto la magnitud como la fase del mensaje con el contexto angular local del átomo receptor i.

**Agregación en espacio cartesiano**:
```
agg_real_i = (1/deg_i) Σ_{j∈N(i)} msg_real_{ij}
agg_imag_i = (1/deg_i) Σ_{j∈N(i)} msg_imag_{ij}
```

La agregación en espacio cartesiano (no polar) corrige el "circular mean problem": si se promedian directamente las fases, dos átomos con fases 0 y π se cancelan, destruyendo información. Al sumar las partes real e imaginaria por separado y reconvertir, la información se preserva correctamente.

### 6.4 Readout Por Átomo

La predicción de la propiedad molecular se realiza mediante readout per-átomo:

```
ε_i = MLP(z_real_i ⊕ z_imag_i) + offset_type(i)
E_mol = (1/N) Σᵢ ε_i
```

donde offset_type(i) ∈ ℝ es un parámetro aprendible por tipo de átomo que codifica la energía atómica de referencia (análogo al "atomic energy baseline" de SchNet y PaiNN). Esto implementa el principio de extensividad: la energía molecular es aproximadamente una suma de contribuciones atómicas más correcciones de interacción.

La media (no la suma) se usa para ser consistente con la normalización per-atom del target en el entrenamiento.

---

## 7. Hipótesis Científica Fortalecida

> **Hipótesis principal:** Un Transformer formulado en el dominio de los números complejos, equipado con una representación polar relacional donde la fase codifica información de orientación relativa a partir de descriptores invariantes a SE(3) (distancias interatómicas y ángulos de enlace), puede aproximar la hipersuperficie de energía potencial molecular con precisión competitiva frente a modelos de segunda generación (SchNet, DimeNet++) y eficiencia computacional significativamente superior a los modelos equivariantes completos (NequIP, Allegro, TensorNet), manteniendo interpretabilidad física de las representaciones intermedias.

**Hipótesis auxiliares:**

**H1** — La separación magnitud-fase en z = r·e^{iθ} implementa una descomposición natural de la información molecular en intensidad de interacción (r) y orientación geométrica relativa (θ), análoga a la descomposición amplitud-fase de los orbitales moleculares.

**H2** — El producto interno hermítico Re(Q·K*) / √d_k en la atención compleja captura tanto la similitud de magnitudes (correlación de intensidades) como la alineación de fases (correlación de orientaciones) en una sola operación, sin aumentar la complejidad paramétrica.

**H3** — La pre-computación de features angulares cos(θ_{jik}) — que son constantes dentro de una molécula — fuera del bucle de capas no solo es un optimización computacional, sino que refleja el principio físico correcto: la geometría molecular no cambia durante el message passing.

**H4** — La regularización de diversidad de fase (minimización de la concentración circular R = |mean(e^{iθ})|) es necesaria para que la representación compleja no colapse a real-valued: sin diversidad de fase, la red pierde la ventaja del dominio complejo.

---

## 8. Comparación con el Estado del Arte

| Modelo | Tipo | Geom. 3D | Invariante | Equiv. SE(3) | Interpretable | MAE U₀ | Eficiencia |
|--------|------|----------|-----------|--------------|---------------|--------|-----------|
| SchNet | GNN radial | Solo dist. | Sí | No | Media | 0.313 | Alta |
| DimeNet++ | GNN angular | Dist+ángulos | Sí | No | Media | 0.215 | Media |
| PaiNN | GNN vectorial | 3D completa | Sí | Parcial | Baja | 0.224 | Media |
| NequIP | Equiv. SE(3) | 3D completa | Sí | Sí | Baja | 0.038 | Baja |
| Allegro | Equiv. SE(3) | 3D completa | Sí | Sí | Muy baja | 0.044 | Muy baja |
| TensorNet | Equiv. SE(3) | 3D completa | Sí | Sí | Muy baja | 0.041 | Muy baja |
| **Este trabajo** | **Transformer complejo** | **Dist+ángulos** | **Sí** | **No** | **Alta** | **~0.5–0.7** | **Media-alta** |

La columna "Interpretable" evalúa si la representación intermedia tiene significado físico directo. La magnitud y la fase del ComplexTensor tienen interpretación directa (intensidad y orientación), a diferencia de los tensores de momento angular irreducibles de NequIP o los tensores de rango 2 de TensorNet.

### 8.1 Ventajas sobre SchNet

SchNet usa solo información radial (distancias). No distingue un ángulo de enlace de 60° de uno de 120° mientras las distancias sean iguales. No puede distinguir el propano del ciclopropano sin ver las distancias de segundo vecino. El modelo propuesto agrega información angular explícita a través de cos(θ_{jik}), codificada en la fase compleja mediante modulación angular.

### 8.2 Ventajas sobre DimeNet++

DimeNet++ usa ángulos de enlace en espacio real, con expansiones de base de Bessel y Fourier. No tiene un mecanismo para representar la *dirección* de la interacción —solo su *magnitud* angular. La representación compleja permite que dos interacciones con el mismo ángulo pero diferente orientación relativa sean distinguidas por su fase, lo que no es posible en representaciones reales.

### 8.3 Ventajas sobre NequIP y modelos equivariantes

Los modelos equivariantes SE(3) son teóricamente más expresivos para propiedades vectoriales y tensoriales (dipolo, polarizabilidad tensorial). Sin embargo:

1. **Complejidad computacional**: las contracciones de Clebsch-Gordan escalan como O(L³) en el grado máximo de momento angular. Para L=3, esto es 27× más operaciones que para L=1.
2. **Sin interpretabilidad**: los tensores de momento angular irreducible (esféricos armónicos de orden 2, 3, 4) no tienen interpretación física directa en el nivel de la representación atómica.
3. **Innecesario para escalares**: para propiedades escalares (energía total, HOMO-LUMO gap), la invariancia es suficiente; la equivariancia completa es un overhead no justificado.

El modelo propuesto ofrece un trade-off diferente: no alcanza la precisión de NequIP, pero es significativamente más interpretable y más eficiente, y ocupa un punto de diseño único en el espacio de modelos moleculares.

---

## 9. El Aporte Diferencial de la Tesis

El aporte científico central de esta tesis es la demostración de que:

> **El dominio complejo polar es un lenguaje natural para la geometría molecular**, en el sentido de que (1) la separación magnitud-fase corresponde a la separación física intensidad-orientación, (2) el producto hermítico implementa atención geométricamente significativa, (3) la modulación de fase por ángulos de enlace propagona información geométrica sin violar invariancias, y (4) la fase acumulada tras el message passing es un resumen interpretable del entorno geométrico molecular local.

Los aportes técnicos derivados son:

**A1 — Representación polar relacional invariante**: embeddings atómicos inicializados desde features químicas y r (distancia al centroide), evolucionados por modulación con descriptores geométricos invariantes (d_{ij}, cos θ_{jik}). Ninguna coordenada no-invariante entra al modelo.

**A2 — Atención compleja hermítica con geometría**: mecanismo de atención basado en Re(Q·K*) que mide simultáneamente similitud de intensidades y alineación de fases, con bias de arista condicionado en RBF para modular la atención por distancia.

**A3 — Message passing complejo con modulación angular precomputada**: paso de mensajes en forma polar con modulación por ángulos de enlace, donde las features angulares se calculan una vez por molécula (no por capa) reflejando el principio físico de que la geometría es constante durante la propagación de información.

**A4 — Readout por átomo con offsets de energía atómica**: predicción extensiva que respeta el principio de descomposición aditiva de la energía molecular en contribuciones atómicas más correcciones de interacción.

**A5 — Regularización de diversidad de fase**: mecanismo novedoso para prevenir el colapso del dominio complejo a real-valued, mediante minimización de la concentración circular de fases.

---

## 10. Dataset, Métricas y Protocolo Experimental

**Dataset**: QM9, ~134.000 moléculas orgánicas (C, H, N, O, F) con geometrías DFT/B3LYP/6-31G(2df,p).

**Split benchmark**: 110.000 entrenamiento / 10.000 validación / resto test. Mismo split que SchNet, DimeNet++, NequIP.

**Propiedad objetivo primaria**: U₀ (energía interna a 0K), referenciada por energías atómicas (u0_atom). Unidades: kcal/mol.

**Métricas**:
- MAE (Error Absoluto Medio): métrica primaria para benchmark QM9
- RMSE: captura outliers
- R²: coeficiente de determinación

**Métricas computacionales** (diferenciadores frente a modelos equivariantes):
- Throughput (muestras/segundo)
- Tiempo por época
- Memoria GPU pico
- Número de parámetros
- Latencia por muestra en inferencia

**Umbral de éxito científico**:
- MAE < 0.7 kcal/mol: mejora significativa sobre SchNet con geometría más simple
- MAE < 0.5 kcal/mol: competitivo con DimeNet++
- MAE < 0.4 kcal/mol: competitivo con PaiNN

---

## 11. Infraestructura Técnica

El proyecto está implementado en Python con PyTorch y ejecutado sobre hardware CUDA.

**Stack principal**: Python, PyTorch, RDKit, NumPy, SciPy.

**Componentes implementados**:

```
ComplexTensor          — z = r·e^{iθ}, operaciones polares
RBFExpansion           — base gaussiana con cosine cutoff
AngularBasis           — base gaussiana sobre ángulos de enlace
ComplexEmbedding       — MLP 2-capas: features → (r, θ)
ComplexPolarAttention  — atención hermítica multi-head con edge-masking
ComplexMessagePassing  — mensaje polar con modulación angular
MagnitudeRMSNorm       — normalización de magnitudes sin discontinuidad en 0
ModReLU                — activación compleja con umbral de magnitud (bias ≤ 0)
ComplexPolarTransformerBeta — modelo completo con readout por átomo
```

**Protocolo de entrenamiento**: AdamW (lr=2e-4, weight_decay=3e-4), cosine annealing con warmup lineal (10 épocas), EMA decay=0.999, AMP (autocast + GradScaler), early stopping sobre val MAE físico (paciencia=40 épocas).

---

## 12. Limitaciones Conocidas y Trabajo Futuro

**Limitación 1 — Invariancia sin equivariancia para propiedades vectoriales**: para propiedades tensorial como la polarizabilidad (tensor 3×3) o el dipolo eléctrico (vector 3D), la invariancia es insuficiente para predecir la dirección del vector en el frame molecular. Extensión natural: implementar una cabeza de predicción equivariante para estas propiedades.

**Limitación 2 — Fase como aproximación de orientación**: la fase compleja captura orientaciones planares (SO(2)), no orientaciones en 3D (SO(3)). Esto es una aproximación del espacio de orientaciones moleculares completo. Extensión natural: usar cuaterniones u octoniones para representar orientaciones en 3D.

**Limitación 3 — Sin diedros**: los ángulos diedros (que determinan conformaciones moleculares, ramificación de cadenas, quiralidad) no están explícitamente modelados. Extensión natural: agregar features de cuadripleta (diedros de enlace) al paso de mensajes.

**Limitación 4 — Geometría fija**: el modelo asume geometría optimizada (equilibrio). Para dinámica molecular o exploración conformacional se necesitaría un modelo que opere sobre geometrías fuera de equilibrio.

---

## 13. Posicionamiento en la Literatura

Esta tesis se ubica en la intersección de tres líneas de investigación activas:

**Geometric Deep Learning para moléculas** (Bronstein et al., 2021; Batzner et al., 2022; Batatia et al., 2022): extensión de GNNs con simetrías físicas.

**Redes neuronales complejas** (Trabelsi et al., 2018; Hirose, 2012): redes neuronales en el dominio complejo con activaciones complejas y álgebra compleja.

**Transformers para química** (Hu et al., 2020; Ying et al., 2021; Liao et al., 2023): aplicación de mecanismos de atención a grafos moleculares y sistemas cuánticos.

El aporte diferencial respecto a estos trabajos es la síntesis: usar el dominio complejo (de las redes complejas) con descriptores relacionales invariantes (del Geometric DL) en una arquitectura Transformer con interpretabilidad física de la fase (nuevo), sin la carga computacional de los modelos equivariantes completos.

---

*Documento generado para uso académico como fundamento científico de tesis de maestría/doctorado en Ciencias Computacionales, línea de Machine Learning aplicado a Química Computacional.*
