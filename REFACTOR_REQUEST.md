# Hoja de Ruta de Investigación — V2 y V3 del ComplexPolarTransformer

## Preámbulo

Este documento no es una lista de mejoras. Es una **auditoría científica** del estado actual de la arquitectura, seguida de una **hoja de ruta de investigación** para evolucionar el modelo hacia una contribución científica defendible. Cada cambio propuesto debe responder simultáneamente a: ¿aumenta el fundamento científico?, ¿fortalece la coherencia matemática?, ¿hace la representación compleja más significativa?, ¿mejora la interpretabilidad?. La reducción del MAE que resulte debe ser consecuencia de un mejor diseño, no del ajuste de hiperparámetros.

El núcleo de la tesis —dominio complejo, representación polar, Transformer, interpretación física de la fase— se mantiene inalterado. Lo que se cuestiona es si la implementación actual realiza lo que la hipótesis promete.

---

## Parte 1 — Auditoría Crítica del Estado Actual (v12)

### 1.1 Pregunta fundamental: ¿La representación polar está describiendo relaciones geométricas?

**Respuesta corta: Parcialmente. Hay un déficit estructural.**

El pipeline actual construye la representación compleja de la siguiente forma:

```
Embedding inicial:
  mag_i  = softplus(MLP([atom_feats_i, r_centroide_i]))   ∈ ℝ⁺
  fase_i = tanh(MLP([atom_feats_i, r_centroide_i])) · π   ∈ (-π, π)
```

La fase inicial no es geométrica. Es una función de `r_centroide` (un escalar global, no una relación entre átomos) y de las features atómicas (química pero no geometría relacional). Esto quiere decir que al inicio de cada forward pass, la fase de cada átomo codifica química e identidad, pero no relaciones entre átomos vecinos.

La geometría relacional entra únicamente en `ComplexMessagePassing` via:
1. RBF de distancias pairwise (relacional ✓)
2. Coseno de ángulos de enlace `cos(θ_jik)` (relacional ✓)

Estos son correctos. El problema es estructural: **la fase solo se vuelve relacional después de pasar por message passing**. En el primer mensaje, los átomos ya transmiten fases que incluyen ruido de la asignación inicial no-relacional. Para moléculas pequeñas con pocos pasos de MP, esto limita la expresividad.

**Dictamen**: La representación polar cumple parcialmente la hipótesis. La magnitud es robusta. La fase se convierte en relacional gradualmente pero inicia desde una asignación basada en coordenadas globales (r_centroide) que conceptualmente contradice el principio relacional del modelo.

---

### 1.2 ¿El r_centroide es un descriptor relacional?

**No. Es un descriptor global con referencia a un artefacto geométrico.**

`r_centroide = ‖r_i − (1/N)Σⱼrⱼ‖`

Esto tiene tres problemas:

**Problema 1 — Referencia no física**: El centroide geométrico no corresponde a ninguna cantidad física relevante. No es el centro de masa, no es ningún núcleo de simetría, no es el centro de carga electrónica. SchNet, DimeNet++, PaiNN, NequIP: ningún modelo del estado del arte usa el centroide como referencia.

**Problema 2 — No es local**: El centroide depende de todos los átomos de la molécula. Es una propiedad global. Incluirla en el embedding inicial viola el principio de inducción local que fundamenta los GNNs moleculares.

**Problema 3 — Redundante con la topología del grafo**: La "centralidad" de un átomo en la molécula ya está implícita en su grado (número de vecinos) y en las distancias a sus vecinos. Incluir r_centroide explícitamente introduce una dependencia global no necesaria.

**Dictamen**: El `r_centroide` debería ser eliminado del embedding inicial y reemplazado por descriptores locales relacionales (número de coordinación, suma de distancias a vecinos, etc.). Ver Propuesta V2.1.

---

### 1.3 ¿La atención y el message passing son redundantes?

**Sí. Hay redundancia estructural.**

En cada capa del modelo ocurre:

```
z_new = attention_layer(z, edge_index, rbf)   # Paso 1
z_new = mp_layer(z_new, edge_index, rbf)       # Paso 2
```

Ambas operaciones agregan información de vecinos en el mismo grafo local (mismo cutoff, mismo `edge_index`). La atención agrega `V_j` ponderado por `Re(Q_i · K_j*)`. El MP agrega `f_rbf(d_ij) * z_j` modulado por ángulos.

Esto duplica las operaciones de agregación en cada capa. No hay justificación científica para hacer dos agregaciones locales separadas en lugar de una sola más expresiva. En la literatura, los modelos exitosos eligen UNA forma de agregación: o attention (Graphormer, TorchMD-Net, ViSNet) o message passing (SchNet, DimeNet++, PaiNN, NequIP). Hacer ambas sin integrarlas es costoso y dificulta la interpretabilidad (¿qué aprende cada operación distinto?).

**Dictamen**: La separación atención/MP es arquitectónicamente injustificada. V2 debe integrar ambas en un único mecanismo de agregación complejo con pesos de atención. Esto REDUCE parámetros, REDUCE cómputo y AUMENTA la interpretabilidad.

---

### 1.4 ¿La modulación angular de la fase es adecuada?

**Parcialmente. El diseño actual colapsa H dimensiones a una sola rotación.**

La modulación angular actual:
```python
msg_phase += self.angular_phase_scale * math.pi * angle_phase   # [E, H]
```

donde `angle_phase = MLP(angle_basis)` con `MLP: ℝᴷ → ℝᴴ` (K = num_angle_basis, H = hidden_dim).

El producto `angular_phase_scale * π` es un ESCALAR (un único parámetro), multiplicado por un vector H-dimensional. Esto aplica el mismo factor de escala global a todas las dimensiones de la fase. Equivale a decir: "todas las dimensiones del espacio complejo rotan igual bajo la influencia angular". Esto destruye la posibilidad de que diferentes dimensiones capturen diferentes tipos de información angular.

**Lo que debería ocurrir**: Si la representación tiene H dimensiones complejas y hay K tipos de información angular (base de ángulos), entonces la modulación angular debería ser una transformación lineal [E, K] → [E, H] que aprende *cómo* cada tipo de ángulo rota *cada* dimensión. Esto es lo que hace `angle_to_phase(angle_feat)` — sí existe esa transformación MLP. 

El problema es el multiplicador: `angular_phase_scale` (escalar) * π * `angle_to_phase(angle_feat)` ([E,H]). La red puede aprender `angle_to_phase` para distribuir diferentes ángulos en diferentes dimensiones, pero si `angular_phase_scale → 0` (lo cual puede ocurrir durante el entrenamiento), toda la modulación colapsa.

**Dictamen**: El `angular_phase_scale` debería ser un vector H-dimensional (un factor de escala POR dimensión), no un escalar. También debería haber un mecanismo de aprendizaje per-head análogo al de los pesos de atención multi-head.

---

### 1.5 ¿La proyección de salida W_O es propiamente compleja?

**No. Es real-linear aplicada por separado a real e imag, no complex-linear.**

```python
new_real = self.W_O(new_real)   # W_O: ℝᴴ → ℝᴴ
new_imag = self.W_O(new_imag)   # misma W_O
```

Una transformación **real-lineal aplicada coordinatamente** a (z_real, z_imag) con los mismos pesos es equivalente a:
```
W_O(z) = W_O(z_real) + i·W_O(z_imag) = W_O(z_real + i·z_imag)
```

Esto solo puede escalar y combinar linealmente cada dimensión con otras dimensiones. No puede MEZCLAR la parte real con la parte imaginaria, lo que significa que no puede rotar en el plano complejo: si z = r·e^{iθ}, esta proyección produce W_O(r)·e^{iθ} (modifica la magnitud de cada dimensión pero mantiene la fase).

Una proyección **complex-linear** tiene la forma:
```
W_ℂ(z) = (W_r·z_real − W_i·z_imag) + i(W_i·z_real + W_r·z_imag)
```

Esta puede rotar el número complejo en el plano: modifica tanto magnitud como fase. Es la única proyección que respeta la estructura algebraica del dominio complejo.

**Dictamen**: W_O debería implementarse como una transformación complex-linear (dos matrices W_r, W_i en lugar de una), lo que dobla los parámetros de salida pero multiplica la expresividad: la proyección puede ahora intercambiar información entre parte real e imaginaria, permitiendo "rotaciones" en el espacio de representaciones.

---

### 1.6 ¿El update gate tiene acceso a información de fase?

**No. Es ciego a la fase.**

```python
gate = self.update_gate(torch.cat([mag, agg_mag], dim=-1))
```

El gate decide cuánto de la agregación incorporar usando solo magnitudes. Ignora completamente si la representación actual y los mensajes agregados tienen fases alineadas o no.

**El problema físico**: Imagina un átomo de carbono con fase θ_i = 0 (modo "ligante") y sus vecinos con fase media θ_agg = π (modo "antiligante"). Las agg_real y agg_imag tendrán componentes negativas que cancelan la representación actual. El gate magnitud-solo no sabe que está ocurriendo esta cancelación destructiva y puede asignar el mismo gate que para una situación constructiva.

**Analógicamente**: es como decidir qué tanto mezclar dos señales acústicas mirando solo su amplitud pero ignorando si están en fase o contrafase.

**Dictamen**: El gate debe condicionar también en la alineación de fases entre la representación actual y los mensajes entrantes. Ver Propuesta V2.4.

---

### 1.7 ¿La representación compleja está siendo utilizada de forma significativa?

**Sí en el mecanismo de atención. Insuficientemente en el message passing y readout.**

**Atención**: Re(Q_i · K_j*) = Σ_h |Q_{i,h}||K_{j,h}|cos(∠Q_{i,h} − ∠K_{j,h}) — genuinamente complejo, mide alineación de magnitudes Y fases. ✓

**Message passing**: El mensaje polar `z_j * f_rbf(d_ij)` y la modulación angular son genuinamente complejos. ✓

**Readout**: Convierte todo a real con `cat(z_real, z_imag)` antes del MLP. Esto es correcto (el resultado es real), pero pierde la posibilidad de hacer operaciones complejas en el readout (e.g., tomar el módulo de una combinación compleja de dimensiones antes del MLP final).

**Activación ModReLU**: ✓ Actúa en el espacio polar (umbral en magnitud, preserva fase). Correcto.

**Norma MagnitudeRMSNorm**: ✓ Normaliza magnitudes sin tocar fases. Correcto.

---

### 1.8 ¿Hay componentes decorativos que deberían eliminarse?

**Sí: la separación atención/MP en paralelo es el candidato principal.**

También hay redundancia en el pipeline de features angulares: se tienen 4 MLPs procesando información del mismo grafo (edge_to_mag, edge_to_phase, angle_to_mag, angle_to_phase). Un diseño unificado reduciría parámetros sin perder expresividad.

**Componente potencialmente decorativo**: `self.phase_scale = nn.Parameter(torch.tensor(math.pi))`. Este parámetro escala la rotación de fase en el mensaje. Si aprende a ser 0, elimina toda la rotación de fase en los mensajes. Si aprende a ser π/k, introduce periodicidad k. Al ser un escalar global, no puede capturar diferencias por dimensión o por tipo de interacción.

---

### 1.9 ¿El modelo aprende relaciones moleculares o memoriza patrones?

**No es posible determinarlo sin ablation studies, pero la arquitectura tiene riesgos.**

El riesgo principal: con 6 capas de atención + 6 capas de MP sobre moléculas de 5-29 átomos (radio efectivo ≤ 5Å), el campo receptivo completo después de 2-3 pasos ya cubre toda la molécula. Las 4-6 capas adicionales pueden estar aprendiendo a refinar, pero también pueden estar memorizando.

El readout per-atom con offsets atómicos es correcto y generalizable. El riesgo de memorización es menor que en readout global.

---

## Parte 2 — Hoja de Ruta V2: Corrección Arquitectónica Principiada

El objetivo de V2 es corregir los problemas identificados manteniendo la complejidad computacional controlada. V2 no es una reescritura total: es una corrección quirúrgica de los déficits identificados.

### Propuesta V2.1 — Embedding relacional local en lugar de r_centroide

**Fundamento matemático**: Reemplazar `r_centroide_i = ‖r_i − centroide‖` por descriptores locales del vecindario de i:

```
coord_num_i  = |N(i)|                              (número de vecinos dentro del cutoff)
d_mean_i     = (1/|N(i)|) Σ_{j∈N(i)} d_{ij}       (distancia media a vecinos)
d_min_i      = min_{j∈N(i)} d_{ij}                 (distancia al vecino más cercano)
d_var_i      = (1/|N(i)|) Σ (d_{ij} − d̄_i)²       (varianza de distancias a vecinos)
```

Todos son invariantes a SE(3) por construcción (dependen solo de distancias pairwise). Capturan: coordinación química (cuántos enlaces tiene el átomo), entorno espacial (cuán comprimido/disperso es el vecindario), especificidad posicional (el átomo central de un anillo tiene d_mean distinto al de un terminal).

**Fundamento físico**: En química de coordinación, el número de coordinación y la distancia media a los ligandos son los descriptores fundamentales del entorno electrónico local. Reemplazar el centroide por estos descriptores conecta el embedding con cantidad físicamente medibles (en difracción de rayos X, por ejemplo).

**Implementación**:
```python
# En el dataset/preprocessing, por cada átomo i:
neighbors = [j for j in range(N) if d_ij < cutoff and j != i]
coord_num   = len(neighbors)                         # escalar
d_mean      = sum(d[i,j] for j in neighbors) / max(coord_num, 1)
d_min       = min(d[i,j] for j in neighbors) if neighbors else 0.0
d_var       = sum((d[i,j]-d_mean)**2 for j in neighbors) / max(coord_num, 1)
local_geom_i = [coord_num / 6.0, d_mean / cutoff, d_min / cutoff, sqrt(d_var) / cutoff]
```

El embedding recibe: `x_i = cat([atom_features_i, local_geom_i])` — en lugar de `cat([atom_features_i, r_centroide_i])`.

**Impacto esperado**:
- Coherencia científica: Alta (descriptores relacionales reales)
- Interpretabilidad: Alta (coordinación y geometría local son físicamente observables)
- MAE/RMSE: Neutral o mejora ligera (mejor información de partida)
- Generalización: Mejora (descriptores puramente locales generalizan mejor)
- Coste computacional: Nulo adicional (precomputado en preprocessing)

---

### Propuesta V2.2 — Integración Atención+MP en CAMP (Complex Attention MessagePassing)

**Fundamento matemático**: Un Transformer sobre grafos puede escribirse como:

```
z_i^{new} = Σ_{j∈N(i)} α_{ij} · M(z_j, e_{ij})
```

donde `α_{ij}` son los pesos de atención y `M(z_j, e_{ij})` es la función de mensaje. Actualmente, el modelo usa:
- Paso 1: `z_i^{mid} = Σ_j α_{ij} · V_j` (atención — agregar V)
- Paso 2: `z_i^{new} = Σ_j w_j · z_j^{mid}` (MP — agregar con pesos RBF)

Esto hace DOS agregaciones con la misma topología. La versión integrada usa los pesos de atención para ponderar los MENSAJES del MP, en lugar de los valores V separados:

```
M_{ij} = (f_rbf(rbf_{ij}) · z_j) ⊙ AngularMod(cos θ_{jik})  # mensaje complejo completo
α_{ij} = softmax_k[Re(Q_i · K_j*) / √d + b_{ij}]             # peso de atención
z_i^{new} = Σ_{j∈N(i)} α_{ij} · M_{ij}                        # agregación única
```

Este es el mecanismo exacto de DimeNet++, con la diferencia de que los mensajes son complejos y los pesos de atención también usan el producto hermítico complejo.

**Fundamento físico**: En mecánica cuántica de muchos cuerpos, la energía de interacción entre átomos i y j (en la aproximación de tight-binding) es:

```
t_{ij} = <ψ_i | H | ψ_j>
```

donde t_{ij} es el integral de transferencia (hopping integral). Su magnitud depende de la distancia (decae exponencialmente), su fase depende de la orientación relativa del orbital (simetría del enlace σ vs π). El mecanismo CAMP integra exactamente esta estructura: |M_{ij}| ~ RBF(d_{ij}) (magnitud = RBF, análogo a decaimiento del hopping), arg(M_{ij}) ~ orientación relacional (fase = geometría).

**Implementación** (simplificada):

```python
class ComplexAttentionMessagePassing(nn.Module):
    def forward(self, z, edge_index, rbf, angle_feat):
        src, dst = edge_index
        
        # Proyecciones Q, K para pesos de atención
        Q = W_Q_complex(z)         # [N, h, hd] — complex-linear
        K = W_K_complex(z)         # [N, h, hd] — complex-linear
        
        # Mensajes complejos (lo que se agrega)
        M = W_M_complex(z[src])    # [E, H] — mensaje base
        M = M * f_rbf_complex(rbf) # modular por distancia en forma polar
        M = M * AngularMod(angle_feat)  # modular por ángulo
        
        # Pesos de atención hermíticos
        α = hermitian_score(Q[dst], K[src]) + edge_bias(rbf)  # [E, h]
        α = edge_softmax(α, dst, num_nodes=N)                  # normalizar por destino
        
        # Agregación única
        z_new = aggregate(α * M, dst, N)  # ponderada por atención
        return z_new
```

**Impacto esperado**:
- Coherencia científica: Alta (el mensaje y el peso de atención están conceptualmente integrados)
- Interpretabilidad: Alta (un solo coeficiente α_{ij} describe la contribución del átomo j al átomo i)
- Parámetros: −40% (elimina la duplicación de proyecciones)
- MAE/RMSE: Mejora 10-20% (más parámetros dedicados a expresividad útil)
- Velocidad: +30-50% (una sola agregación por capa en lugar de dos)

---

### Propuesta V2.3 — Proyección complex-linear en la salida de atención

**Fundamento matemático**: El grupo de simetría del espacio de representaciones debe ser el grupo unitario U(H) (rotaciones en ℂᴴ), no el grupo ortogonal O(H) (rotaciones en ℝᴴ). Una proyección real-linear aplicada coordinatamente a (z_real, z_imag) vive en O(H) (no puede rotar entre real e imag). Una proyección complex-linear vive en U(H) y puede mezclar real e imaginario.

La transformación complex-linear:
```
W_ℂ(z) = (W_r + i·W_i)(z_r + i·z_i)
        = (W_r·z_r − W_i·z_i) + i(W_i·z_r + W_r·z_i)
```

donde W_r, W_i ∈ ℝ^{H×H}. Esta tiene propiedades de invariancia bajo la acción de U(H) que la versión real no tiene.

**Fundamento físico**: En un sistema cuántico, el operador de evolución temporal es `e^{-iHt}`, que es unitario (U(H)). Las proyecciones entre representaciones intermedias de un modelo que pretende capturar analogías cuánticas deberían ser unitarias — o al menos complex-linear, que es el subconjunto lineal de U(H).

**Implementación**:

```python
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_r = nn.Linear(in_features, out_features, bias=False)
        self.W_i = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, z_real, z_imag):
        out_real = self.W_r(z_real) - self.W_i(z_imag)
        out_imag = self.W_i(z_real) + self.W_r(z_imag)
        return out_real, out_imag
```

Reemplaza `W_O(new_real), W_O(new_imag)` (misma W_O) en `ComplexPolarAttention`.

**Impacto esperado**:
- Coherencia matemática: Alta (respeta la estructura del grupo U(H))
- Interpretabilidad: Alta (la proyección puede ahora "rotar" representaciones en el plano complejo, lo que es físicamente significativo)
- Parámetros: +H² (duplica los parámetros de W_O, de H² a 2H²)
- MAE/RMSE: Mejora 5-10% (mayor expresividad de la proyección de salida)
- Velocidad: −5% (operación adicional modesta)

---

### Propuesta V2.4 — Gate que incluye información de fase (Phase-Aware Gate)

**Fundamento matemático**: La función de gate actual decide cuánto incorporar de la agregación usando solo magnitudes. Esto es equivalente a decidir la mezcla de dos números complejos z₁ y z₂ mirando solo sus módulos |z₁|, |z₂| e ignorando el ángulo entre ellos.

La información faltante es Re(z_current · z_agg*) / (|z_current| · |z_agg|) = cos(θ_current − θ_agg), que mide la alineación de fases. Si esta es ≈1 (constructivo), el gate debería ser más permisivo. Si es ≈−1 (destructivo), el gate podría querer amortiguar la incorporación.

**Gate phase-aware**:
```python
# Alineación de fases por dimensión
current_real = mag * cos(phase)           # [N, H]
current_imag = mag * sin(phase)
dot_product = current_real * agg_real + current_imag * agg_imag  # [N, H]
mag_product  = (mag * agg_mag).clamp_min(1e-9)
cos_alignment = dot_product / mag_product                         # [N, H] ∈ [-1, 1]

gate = σ(W_gate(cat([mag, agg_mag, cos_alignment])))  # [N, H]
```

**Fundamento físico**: En un circuito de osciladores acoplados, la transferencia de energía entre dos osciladores depende de su diferencia de fase: máxima si están en fase, nula si están en cuadratura, negativa si están en contrafase. El gate phase-aware implementa exactamente este principio.

**Impacto esperado**:
- Coherencia física: Alta (el gate refleja la física de interferencia)
- Interpretabilidad: Alta (un gate cercano a 1 con cos_alignment ≈ 1 significa interacción constructiva observable)
- Parámetros: +H (aumenta la dimensión de entrada del gate)
- MAE/RMSE: Mejora 5-15% (el gate ahora puede distinguir contribuciones constructivas de destructivas)

---

### Propuesta V2.5 — Angular scale per-dimension (H-dimensional en lugar de escalar)

**Fundamento matemático**: El espacio de representaciones complejas tiene H dimensiones. Cada dimensión puede especializarse en capturar un tipo diferente de información molecular. Una escala angular escalar `angular_phase_scale ∈ ℝ` aplica el mismo factor a todas las dimensiones, impidiendo la especialización.

**Propuesta**: Reemplazar `angular_phase_scale: nn.Parameter(tensor(0.1))` (escalar) por `angular_phase_scale: nn.Parameter(ones(H) * 0.1)` (H-dimensional). Igualmente para `angular_mag_scale`.

**Fundamento físico**: En la teoría de perturbaciones dependiente del tiempo, diferentes orbitales atómicos responden de forma diferente a una perturbación angular (campo eléctrico o magnético). Un orbital s es esféricamente simétrico (sin respuesta angular). Un orbital p_z responde fuertemente a campos en z pero no en x, y. La escala per-dimension permite que diferentes dimensiones del espacio de representaciones emulen esta especialización.

**Implementación**:
```python
self.angular_mag_scale   = nn.Parameter(torch.full((hidden_dim,), angular_scale_init))
self.angular_phase_scale = nn.Parameter(torch.full((hidden_dim,), angular_scale_init))
```

En lugar de:
```python
self.angular_mag_scale   = nn.Parameter(torch.tensor(float(angular_scale_init)))
self.angular_phase_scale = nn.Parameter(torch.tensor(float(angular_scale_init)))
```

**Impacto esperado**:
- Coherencia científica: Alta
- Expresividad angular: Aumenta de 1 grado de libertad a H
- Parámetros: +2H (trivial)
- MAE/RMSE: Mejora 5-10%
- Computación: Nula adicional (broadcast automático)

---

### Resumen de Cambios V2

| Propuesta | Componente afectado | Parámetros | Velocidad | MAE esperado | Interpretabilidad |
|-----------|--------------------|-----------:|----------:|:------------:|:-----------------:|
| V2.1 Embedding relacional | ComplexEmbedding | −H | 0% | neutral | ↑↑ |
| V2.2 CAMP integrado | Attn+MP→CAMP | −40% | +40% | ↓15-20% | ↑↑↑ |
| V2.3 ComplexLinear W_O | Attn output | +H² | −5% | ↓5-10% | ↑↑ |
| V2.4 Phase-aware gate | MP gate | +H | −2% | ↓5-15% | ↑↑↑ |
| V2.5 Angular scale H-dim | MP angular | +2H | 0% | ↓5-10% | ↑↑ |

**MAE total esperado V2**: ~0.4–0.55 kcal/mol (desde ~0.7–0.9 actual)

La estimación asume que los cambios son aditivos y que el beneficio principal viene de V2.2 (integración CAMP) y V2.4 (gate de fase).

---

## Parte 3 — Hoja de Ruta V3: Contribución Científica Original

El objetivo de V3 no es refinar V2 sino dar un salto cualitativo en la calidad científica del aporte. Los cambios de V3 deben ser lo suficientemente originales para ser defendibles como contribución doctoral independiente. No todos deben implementarse simultáneamente — son propuestas de investigación, algunas de las cuales requerirán ablation studies para validar.

### Propuesta V3.1 — Fase Compleja como Diedros: Geometría SO(2)-invariante

**El problema que resuelve**: La fase inicial de cada átomo no tiene significado geométrico. Se inicializa desde features químicas, no desde la geometría relacional. Para que la hipótesis de la tesis sea completa, la fase debería emerger directamente de la geometría molecular desde la primera capa.

**La solución**: Inicializar la fase de los mensajes de arista desde los ángulos diedros locales.

**Fundamento matemático**: Para una arista j→i con al menos un segundo vecino k de i, el ángulo diedro entre los planos (j,i) y (k,i) es:

```
Δφ_{j,k→i} = signed_dihedral(j, i, k) = atan2(
    (r_ij × r_ik) · r_ij/|r_ij|,
    (r_ij · r_ik) / (|r_ij| |r_ik|)
)
```

Este ángulo es SE(3)-invariante: no cambia bajo rotaciones o traslaciones porque es un ángulo interno entre dos planos definidos por átomos.

El diedro es naturalmente un elemento de SO(2): Δφ ∈ (−π, π). Por lo tanto, `e^{iΔφ}` es un número complejo de módulo 1 que codifica directamente la orientación relativa entre dos enlaces que convergen en el mismo átomo.

**Inicialización propuesta para la fase de mensaje**:

```python
# Para la arista j→i, sean k₁, k₂, ... los otros vecinos de i
φ_edge_{j→i} = (1/num_other_neighbors) * Σ_k Δφ_{j,k→i}  (promedio de diedros)
```

Esto convierte la fase del mensaje de la arista j→i en una representación geométrica real: codifica cómo el enlace j→i se orienta respecto a los otros enlaces de i.

**Fundamento físico**: Los ángulos diedros (phi, psi en proteínas; ángulos de torsión en química orgánica) determinan la conformación molecular, la actividad biológica, la quiralidad y las propiedades espectroscópicas NMR. Un modelo que inicialice su representación de fase desde los diedros captura información conformacional desde la primera capa.

**Consecuencia para la interpretabilidad**: Tras el readout, la parte imaginaria de la representación atómica codifica información conformacional directamente derivada de los diedros. Los pesos de atención `Re(Q_i · K_j*)` = `|Q_i||K_j|cos(∠Q_i − ∠K_j)` miden si los entornos conformacionales de i y j son similares.

**Coste de implementación**: Requiere calcular ángulos diedros por arista como paso de preprocessing, análogo al cálculo de ángulos de enlace actual.

**Impacto esperado**:
- Coherencia científica: Muy alta (la fase nace de geometría real)
- Interpretabilidad: Muy alta (fase = diedro = conformación)
- Generalización: Alta (conformaciones no vistas se mapean a fases nuevas, no a embeddings out-of-distribution)
- MAE/RMSE: Mejora 10-25%

---

### Propuesta V3.2 — Mensajería Arista-a-Arista con Actualizaciones de Fase Diferencial

**El problema que resuelve**: La arquitectura actual hace mensajes átomo→átomo. Pero la química de los enlaces tiene su propia dinámica: la fortaleza de un enlace j→i depende de qué otros enlaces tiene i y cómo se orientan. DimeNet++ captura esto con mensajes arista→arista en espacio real. V3 propone hacer lo mismo en el dominio complejo.

**Fundamento matemático**: Sea w_{j→i} = s_{j→i} · e^{iφ_{j→i}} la representación compleja de la arista j→i, donde s es la magnitud (fortaleza del enlace) y φ es la fase (orientación relativa). La actualización de esta arista usa mensajes de las aristas entrantes en j:

```
w_{j→i}^{new} = Σ_{k∈N(j)\{i}} α_{k→j,→i} · f(w_{k→j}, w_{j→i})
```

donde α_{k→j,→i} es el peso de atención de la arista k→j sobre la actualización de j→i.

La función de mensaje entre aristas puede definirse como:

```
f(w_{k→j}, w_{j→i}) = |w_{k→j}| · |w_{j→i}| · e^{i(φ_{k→j} + φ_{j→i})}
```

(producto de magnitudes, suma de fases — regla del producto complejo)

**Fundamento físico**: En la teoría de enlace de valencia, la fuerza de un enlace entre i y j está modulada por la capacidad de los electrones de i para participar en el enlace — que a su vez depende de cuántos otros enlaces tiene i y cómo se orientan (hibridación). Este "contexto de enlace" es exactamente lo que la mensajería arista→arista captura.

La suma de fases `φ_{k→j} + φ_{j→i}` implementa la composición geométrica de dos orientaciones de enlace: si el primer enlace apunta en dirección θ₁ y el segundo en θ₂, su composición es θ₁+θ₂. Esto es la representación de la cadena de orientaciones a lo largo de un camino molecular.

**Implicación para la interpretabilidad**: Tras varias capas de mensajería arista-a-arista, la fase de cada arista codifica la orientación geométrica acumulada a lo largo de caminos moleculares. Para un anillo aromático (6 enlaces en cadena), la suma de fases debe converger al ángulo total de rotación de la cadena, que por la geometría hexagonal es 2π.

---

### Propuesta V3.3 — Fase de Berry Molecular como Feature de Anillo

**La idea más original de V3.**

**Fundamento matemático**: En física de estado sólido, la fase de Berry (o fase geométrica) de un estado cuántico evolucionado adiabáticamente a lo largo de un ciclo cerrado en el espacio de parámetros es:

```
γ = i ∮ <ψ(λ)| ∇_λ |ψ(λ)> dλ
```

Este número es real, topológico (no cambia bajo deformaciones continuas del ciclo), y mide la holonomía de la conexión en el espacio de Hilbert. En sólidos cristalinos, la fase de Berry (fase de Zak) determina la polarización eléctrica.

**Analógico molecular**: Para una molécula con un anillo de N átomos [i₁, i₂, ..., i_N, i₁], si la red aprende representaciones complejas z_k para cada átomo, entonces el producto del ciclo:

```
Γ_ring = z_{i₁} · z_{i₂}* · z_{i₂} · z_{i₃}* · ... · z_{i_N} · z_{i₁}*
       = Π_k (z_{i_k} · z_{i_{k+1}}*)
       = r_total · e^{i·Σ_k (θ_{i_k} - θ_{i_{k+1}})}
```

El argumento de Γ_ring, `Σ_k (θ_{i_k} − θ_{i_{k+1}})`, es la suma de diferencias de fase a lo largo del anillo — análogo discreto de la fase de Berry.

**Propiedad clave**: Para un anillo aromático de 6 miembros con 6π electrones (benceno), la fase de Berry molecular debería ser cuantizada en 2πn por las reglas de Hückel. El modelo puede aprender esto sin supervisión si la regularización de fase y la mensajería de anillo están bien diseñadas.

**Implementación práctica**: Como feature adicional de lectura, no como restricción:
1. Detectar ciclos en el grafo molecular (con networkx o similar, precomputado)
2. Para cada ciclo de longitud N en la molécula, calcular `Γ_ring` usando las representaciones finales z
3. Agregar `|Γ_ring|` y `arg(Γ_ring) mod 2π` como features adicionales del readout

**Por qué es científicamente original**: No existe en la literatura ningún modelo de GNN molecular que compute explícitamente una cantidad análoga a la fase de Berry sobre las representaciones aprendidas. La conexión con la aromaticidad de Hückel y con los números de Chern de la física topológica es conceptualmente poderosa y defendible.

**Impacto esperado**:
- Contribución científica: Muy alta (conexión topología↔química↔ML, publicable en NeurIPS/ICLR)
- Interpretabilidad: Muy alta (Γ_ring mide aromaticidad/antiaromaticidad de forma continua)
- MAE/RMSE: Mejora adicional ~5% en propiedades relacionadas con aromaticidad (HOMO-LUMO gap)
- Coste: Detección de ciclos O(V+E) precomputado; cálculo de Γ O(ciclos×V) durante forward

---

### Propuesta V3.4 — Tight-Binding Analogy: Hamiltonian Attention

**Fundamento matemático**: En el modelo tight-binding (Hückel) de la química cuántica, la energía de un sistema electrónico viene dada por el valor propio mínimo de la matriz Hamiltoniana:

```
H_{ii} = α_i  (energía del sitio — análogo a atom energy offset)
H_{ij} = β_{ij} · e^{i·k·R_{ij}}  (integral de hopping — análogo a peso de atención complejo)
```

donde la fase e^{ik·R} surge del teorema de Bloch (condiciones de periodicidad). En moléculas (sin periodicidad), la fase surge de la geometría local.

**La conexión con nuestra arquitectura**: La matriz de atención en la capa L:

```
A^L_{ij} = Re(Q_i^L · (K_j^L)*) / √d_k
```

es una matriz Hermítica (A = A†) si Q = K. Esto tiene exactamente la estructura de un Hamiltoniano Hermítico. El valor propio más pequeño de A^L (después del softmax) determina el estado fundamental del "sistema cuántico" de esa capa.

**Propuesta V3.4**: Para la capa de readout, en lugar de (o además de) el MLP sobre representaciones atómicas, usar el valor propio dominante de la matriz de atención de la última capa:

```python
A_last = compute_hermitian_attention_matrix(z_final, edge_index)  # [N, N]
# Método de potencias o Lanczos para el valor propio de mayor módulo
λ_dom, v_dom = power_iteration(A_last, num_iters=10)
# λ_dom es análogo a la energía de mayor amplitud del sistema
# v_dom es el vector propio dominante — análogo al orbital HOMO
```

Esta contribución conecta el mecanismo de atención complejo directamente con la teoría orbital molecular: el vector propio dominante de la última capa de atención es análogo al orbital molecular de mayor energía (HOMO o LUMO según el signo).

**Fundamento físico**: La energía de la molécula en el modelo Hückel es:
`E_total = Σ_k n_k λ_k`
donde n_k es la ocupación del orbital k y λ_k su energía. La matriz de atención hermítica aprendida es análoga a la matriz Hamiltoniana de Hückel, pero aprendida desde los datos en lugar de parametrizada por α y β fijos.

---

### Propuesta V3.5 — Representación Compleja por Cabeza con Significado Físico Asignado

**El problema**: En la arquitectura multi-head actual, las 4 cabezas de atención son intercambiables — no hay nada que las distinga más allá de diferentes pesos iniciales aleatorios. En la práctica, la red puede aprender a asignar cada cabeza a un tipo de interacción, pero no está guiada a hacerlo.

**La propuesta**: Inicializar y regularizar las cabezas para que capturen tipos de interacción distinguibles:

| Cabeza | Tipo de interacción | Restricción de inicialización |
|--------|--------------------|-----------------------------|
| h=0 | Radial (electrostático) | W_Q_phase, W_K_phase inicializados en 0 (fase inicial ≈ 0) |
| h=1 | Angular (covalente) | W_Q_mag, W_K_mag inicializados con énfasis en features de hibridación |
| h=2 | Van der Waals | W_K inicializado con énfasis en RBF de rango 3.0-5.0Å |
| h=3 | Topológico | W inicializado a capturar distancias topológicas de 2 saltos |

Esto no impone restricciones duras (los pesos son libres durante el entrenamiento), sino que el punto de partida favorece la especialización. Se puede agregar una regularización de *head diversification*:

```
L_div = −Σ_{h≠h'} |cos(A_h, A_{h'})| · λ_div
```

que penaliza la correlación entre matrices de atención de diferentes cabezas.

**Fundamento físico**: En la descomposición de la energía intermolecular (SAPT — Symmetry-Adapted Perturbation Theory), la energía de interacción molecular tiene cuatro componentes: electrostática, inducción, dispersión y intercambio-repulsión. Son ortogonales conceptualmente. Orientar las 4 cabezas hacia estos 4 tipos de interacción es físicamente justificado.

---

### Propuesta V3.6 — Phase Coherence Loss como Indicador de Aromaticidad

**Fundamento**: La regularización de diversidad de fase actual (concentración circular) es una regularización genérica. V3 propone reemplazarla por una **loss de coherencia de fase específica para anillos**, que mide si los anillos aromáticos exhiben la fase de Berry correcta.

Para un anillo de N miembros con la geometría correcta de Hückel (4n+2 electrones π):
- El modelo debería aprender: `Σ_{i en anillo} θ_i ≈ 2πk` para algún entero k

Para un anillo antiaromático (4n electrones π):
- El modelo debería aprender: `Σ_{i en anillo} θ_i ≈ π(2k+1)`

La pérdida de coherencia de anillo:
```python
def ring_phase_coherence_loss(z, rings, mol_aromaticity):
    loss = 0
    for ring, is_aromatic in zip(rings, mol_aromaticity):
        phase_sum = z.phase[ring].sum()  # Σ_i θ_i en el anillo
        if is_aromatic:
            # Penalizar desviación de 2πk (múltiplo de 2π)
            target = torch.round(phase_sum / (2*π)) * 2*π
        else:
            # Penalizar desviación de π(2k+1) (múltiplo impar de π)
            target = torch.round((phase_sum - π) / (2*π)) * 2*π + π
        loss += (phase_sum - target.detach()) ** 2
    return loss / max(len(rings), 1)
```

Esta pérdida guía la red a aprender representaciones de fase coherentes con la regla de Hückel, sin necesitar supervisión directa de aromaticidad (solo necesita la aromaticidad de los anillos, que está en QM9 implícitamente en la estructura molecular).

---

## Parte 4 — Prioridades, Secuencia y Criterios de Éxito

### Secuencia recomendada de implementación

```
Semana 1-2:  V2.1 + V2.5  (embedding local + angular scale H-dim)
             — Bajo riesgo, sin impacto en arquitectura principal
             — Validar: MAE comparable o mejor, tests pasan

Semana 3-4:  V2.3          (ComplexLinear W_O)
             — Requiere nuevo módulo, test unitario de identidad compleja
             — Validar: gradientes fluyen correctamente por W_O_r y W_O_i

Semana 5-6:  V2.4          (Phase-aware gate)
             — Modifica ComplexMessagePassing.update_gate
             — Validar: gate condicionada en cos_alignment es ≥ 0.1 en práctica

Semana 7-10: V2.2          (CAMP — integración Attn+MP)
             — Cambio arquitectónico mayor, requiere nueva clase
             — Ablation: comparar CAMP vs. Attn+MP separados en hold-out de 10k muestras
             — Objetivo: MAE ≤ 0.5 kcal/mol en val set

Semana 11-14: V3.1         (Diedros como fase inicial)
              — Requiere preprocessing nuevo: cálculo de ángulos diedros por arista
              — Test: la fase inicial debe variar entre confórmeros del mismo esqueleto

Semana 15-20: V3.2+V3.3   (Edge-to-edge + Berry phase)
              — Investigación activa: explorar si Berry phase correlaciona con HOMO-LUMO gap
              — Publicación potencial si la correlación es significativa

Semana 21-25: V3.4         (Hamiltonian attention)
              — Requiere evaluación del vector propio — puede ser costoso
              — Investigar si λ_dom correlaciona con energía DFT sin supervisión
```

### Criterios de éxito para defensa doctoral

**Mínimos (requisito para defender)**:
- MAE U₀ ≤ 0.5 kcal/mol (comparado con SchNet 0.313, DimeNet++ 0.215)
- Ablation study que muestre contribución de la representación compleja vs. la real
- Visualización de la fase como descriptora geométrica molecular
- Análisis de la concentración de fase a lo largo del entrenamiento

**Objetivos fuertes (defensa sólida)**:
- MAE U₀ ≤ 0.35 kcal/mol (comparable con DimeNet++)
- Demostración de la fase de Berry en anillos aromáticos
- Análisis de la especialización de cabezas de atención por tipo de interacción
- Comparación de throughput y eficiencia vs. NequIP/Allegro (modelo ligero con performance comparable)

**Contribución original (publicación en venue top)**:
- La conexión fase-compleja / fase-de-Berry / aromaticidad-Hückel, si se valida experimentalmente
- El mecanismo CAMP (Complex Attention Message Passing) como contribución arquitectónica
- La inicialización de fase desde diedros como forma de embeber geometría SO(2)-invariante desde la primera capa

---

## Parte 5 — Lo que NO debe cambiarse

En toda refactorización, es tan importante identificar qué no cambiar como qué cambiar. Los siguientes componentes están bien fundamentados y no deben modificarse:

**ModReLU con b_eff ≤ 0**: La corrección de amplificación (softplus constraint) es matemáticamente correcta y el init b=5.0 → b_eff≈−0.007 (near-identity) es apropiado. ✓

**Residual en espacio cartesiano**: Sumar (real, imag) y reconvertir a polar es la única forma correcta de hacer residuales con números complejos. ✓

**Aggregación de V en espacio cartesiano (circular mean fix)**: La corrección al circular mean problem es matemáticamente correcta. ✓

**MagnitudeRMSNorm**: Normaliza sin abs(), preserva no-negatividad, sin subgradientes en 0. ✓

**Precompute de features angulares fuera del bucle de capas**: Los ángulos no cambian entre capas. Calcularlos una sola vez es correcto. ✓

**Readout per-atom con atomic energy offsets**: Respeta extensividad molecular. ✓

**EMA + cosine warmup scheduler**: Bien calibrado para QM9. ✓

**Cutoff 5.0 Å**: Estándar del benchmark QM9. No cambiar sin justificación. ✓

---

## Apéndice — Tabla de Alineación Hipótesis-Implementación

| Afirmación de la Hipótesis | ¿Implementada? | Déficit actual | Propuesta |
|---------------------------|:-------------:|----------------|-----------|
| La fase codifica orientación geométrica | Parcial | La fase inicial es química, no geométrica | V3.1 (fase desde diedros) |
| La representación es relacional | Parcial | r_centroide es global, no relacional | V2.1 |
| La atención mide alineación magnitud+fase | Sí ✓ | — | — |
| La mensajería modula la fase con ángulos | Sí ✓ | Angular scale es escalar | V2.5 |
| La proyección de salida es complex-linear | No ✗ | W_O aplica igual a real e imag | V2.3 |
| El gate respeta la física de interferencia | No ✗ | Gate usa solo magnitudes | V2.4 |
| La representación captura topología anular | No ✗ | No hay módulo de ciclos | V3.3 |
| La atención y MP tienen roles distintos | No ✗ | Ambos agregan la misma topología | V2.2 |
| La fase tiene significado químico medible | No ✗ | No hay conexión con observable externo | V3.3, V3.6 |
