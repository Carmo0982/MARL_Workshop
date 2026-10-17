# MARL Stock Exchange — Informe / Evidencia experimental

**Equipo:** TEAM_19 — Risk-Aware Bulls

**Integrantes:** Santiago Carmona, Nicole Calderón

**Modelo entregado:** `ft_abides25` (Double Dueling DQN con n-step, afinado con datos reales de ABIDES-NG)

---

## 1. Algoritmo y arquitectura

**Algoritmo:** Double DQN con retornos n-step (n = 3), actualización suave de la red objetivo (Polyak, τ = 0,005) y exploración ε-greedy (ε de 1,0 a 0,02 durante el primer 40 % del entrenamiento).

**Arquitectura** (`student_work/model.py::DuelingQNetwork`):

```
16 → Linear(128) → LayerNorm → ReLU → Linear(128) → ReLU ─┬─ V(s):   Linear(64) → ReLU → Linear(1)
                                                           └─ A(s,a): Linear(64) → ReLU → Linear(9)
Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)
```

| Decisión | Por qué |
|---|---|
| **Double DQN** | `max_a Q_target` sobreestima los valores cuando las recompensas son ruidosas, y un mercado es muy ruidoso. Elegir la acción con la red online y evaluarla con la red objetivo reduce ese sesgo. |
| **Dueling** | En la mayoría de los estados casi no importa qué acción se toma. Separar V(s) permite aprender el valor del estado con cada transición, no solo con la acción ejecutada. |
| **LayerNorm** | Estabiliza las activaciones entre regímenes muy distintos (estable frente a crisis) y en inferencia no depende de estadísticas acumuladas, así que el modelo es seguro para TorchScript en CPU. |
| **n-step = 3** | Propaga más rápido las ganancias y pérdidas de una posición, que se realizan varios pasos después de abrirla. |
| **DQN y no PPO** | El espacio de acciones es discreto y pequeño (9 acciones), y DQN es off-policy: puede reutilizar transiciones de cualquier política, incluidas las recolectadas en ABIDES (sección 6). |
| **Fine-tuning** | El modelo entregado parte de `v2_dueling_s7` y se afinó 60.000 pasos más con 6.581 transiciones reales recolectadas en ABIDES-NG, que ocupan el 25 % de cada batch del replay (lr 2,5e-4; ε de 0,2 a 0,02). La arquitectura y el contrato (`[batch,16] → [batch,9]`, TorchScript, CPU) no cambian. |

El modelo exportado cumple el contrato: entrada float32 `[batch,16]`, salida float32 `[batch,9]`, TorchScript, CPU, valores finitos. `train.export_torchscript` lo comprueba tras cada guardado.

## 2. Función de recompensa

El starter usa `r = ΔNAV/1000 − 0,0005·|inv|`. Nosotros usamos `RewardConfig.risk_aware()` (`student_work/envs.py`):

```
r_t = ΔNAV_t / 1000                      (1 unidad = $10)
      − 0.001 · |inventario_t|           (riesgo de inventario)
      − 0.5 · max(0, −ΔNAV_t) / 1000     (pérdidas del paso: una pérdida pesa 1,5× una ganancia)
      − 0.02 · (# órdenes ejecutadas)    (costo de rotación / anti-churn)
      − 0.01 · |inventario_final|        (al terminar el episodio: penaliza dejar posición abierta)
```

**Por qué:** con el reward del starter, la penalización por inventario es despreciable frente al ruido del PnL. Con 500 acciones resta 0,25 por paso, mientras que el ruido de un paso es de unos ±3,5. El agente puede entonces aprender exposiciones extremas que ganan en promedio pero tienen drawdowns enormes.

El término de "drawdown" se implementó como incremento del drawdown respecto al máximo de NAV, pero algebraicamente equivale a `max(0, −ΔNAV_t)`: penaliza **cada pérdida por paso**, no solo los nuevos máximos de drawdown (lo detectó la revisión externa). En la práctica es una utilidad asimétrica: las pérdidas pesan 1,5× las ganancias, lo que castiga la varianza a la baja. Todos los modelos se entrenaron así; lo dejamos documentado en vez de cambiarlo. Las penalizaciones son pequeñas a propósito, para que HOLD no se vuelva la política óptima trivial.

**Ablación:** el mismo algoritmo entrenado con el reward del starter (`ddqn_starter_reward`) obtiene más retorno y tiene un drawdown unas 1,5 veces mayor en simulación; en los mercados normales de ABIDES su drawdown es unas 4–5 veces mayor y acumula más inventario. Ver la sección 5.

## 3. Escenarios de entrenamiento y semillas

**Entorno:** `student_work/envs.py::RobustMarketEnv`. Usa el mismo `build_observation` y el mismo `action_to_order` del contrato oficial (no modificamos `shared/`). Añade:

- **12 regímenes:** stable, trend_up, trend_down, volatile, mean_revert, low_liquidity, stress (crisis), crowded (otros seguidores de tendencia) y 4 regímenes **calibrados contra ABIDES-NG** (abides_calm, abides_trending, abides_volatile, abides_stress).
- **Aleatorización de dominio:** en cada episodio se elige un régimen, se perturban sus parámetros ±30 %, el precio inicial (entre $50 y $1.500, escala logarítmica) y la duración (200–400 pasos). El 15 % de los episodios usa el `PracticeMarketEnv` oficial.
- **Órdenes limitadas que quedan en el libro** con un tiempo de vida (5 pasos), así `open_orders` deja de ser siempre 0, igual que en ABIDES.
- **Impacto de mercado** para órdenes grandes en libros delgados, y spread que se amplía tras movimientos fuertes.

**Separación de semillas:**

| Uso | Semillas |
|---|---|
| Entrenamiento | `seed·100000 + episodio` (distintas en cada episodio) |
| Validación, para elegir el checkpoint | 1001, 1002, 1003 |
| Test final (usar **una sola vez**) | 5001–5005 |
| ABIDES, evaluación | 1, 2 (y 3–5 para el test final) |
| ABIDES, recolección de datos | 11–14 |

## 4. Hiperparámetros importantes

| Parámetro | Valor |
|---|---|
| Pasos de entrenamiento | 200.000 (v2) |
| Optimizador | Adam, lr 5e-4 con decaimiento lineal hasta el 10 % |
| Batch / replay | 128 / 200.000 |
| γ | 0,99 |
| n-step | 3 |
| τ (actualización suave) | 0,005 |
| ε | 1,0 → 0,02 en el 40 % de los pasos |
| Recorte de gradiente | 10 |
| Pérdida | Huber (smooth L1) |
| Peso de los escenarios ABIDES en el muestreo | 2× |
| Evaluación de validación | cada 10.000 pasos, suite completa (12 escenarios × 3 semillas) |
| Precio en evaluación | fijo en $100 (en entrenamiento, aleatorio entre $50 y $1.500) |
| Fine-tuning (modelo entregado) | 60.000 pasos, lr 2,5e-4, ε 0,2→0,02, 25 % de datos de ABIDES por batch |

La configuración exacta de cada corrida queda en `outputs/runs/<run>/config.json`.

## 5. Resultados (semillas de validación)

### 5.1 Comparaciones entre candidatos

Simulación: `outputs/eval/v3_full_fixed/` (evaluador corregido: semilla propia por escenario y precio fijo en $100).
ABIDES: `outputs/abides_eval/v2/summary.md` (5 escenarios × 2 semillas). "maxDD" en ABIDES excluye `stress_test`.

**Ninguna de estas comparaciones es una ablación limpia**: cada una cambia varias cosas a la vez, así que las
diferencias no se pueden atribuir a un solo factor.

| Comparación | Qué cambia realmente | Simulación: score / maxDD medio / peor escenario | ABIDES: media sin stress / maxDD sin stress |
|---|---|---|---|
| **Recompensa:** risk_aware vs starter (v1) | solo la recompensa | −0,21 / 0,83 % / −0,57 % **vs** 0,41 / 1,24 % / −0,44 % | −230 / $478 **vs** −345 / $2.220 |
| **Paquete algorítmico:** v2 Dueling vs v2 MLP (semilla 7) | Dueling + LayerNorm + 128 unidades + Double DQN + n-step 3 **vs** MLP 64 sin LayerNorm + DQN + n-step 1 | 0,56 / 0,42 % / −0,41 % **vs** 0,00 / 1,03 % / −2,22 % | −318 / $1.599 **vs** −741 / $1.601 |
| **v2 vs v1** | +4 escenarios `abides_*`, otro `stress`, 200k vs 150k pasos (y por tanto otro calendario de ε y lr), checkpoint elegido con 12 vs 8 escenarios | 0,40 (s11), 0,56 (s7) **vs** −0,21 | +22 (s11), −318 (s7) **vs** −230 |
| **Fine-tuning con datos de ABIDES** (25 % / 50 %) vs `v2_dueling_s7` | datos de ABIDES **y** +60k pasos con lr 2,5e-4, ε 0,2→0,02; no hay control con +60k pasos sin datos de ABIDES | 0,30 / 0,62 **vs** 0,56 | +234 / −490 **vs** −318 (maxDD $355 / $910 vs $1.599) |
| **Semilla:** v2 s7 vs s11 | solo la semilla | 0,56 **vs** 0,40 | −318 **vs** +22 |

Lectura (con la cautela anterior):
1. En simulación todos los candidatos RL quedan cerca entre sí y muy por encima del DQN starter (−2,99). La
   variación entre semillas (0,40 frente a 0,56) es del mismo orden que las diferencias entre configuraciones.
2. La recompensa con riesgo **no** mejora el score en simulación, pero en ABIDES reduce unas 4–5 veces el drawdown
   en mercados normales y la exposición (inventario máximo medio 210 frente a 468).
3. El paquete Dueling/Double/n-step reduce el riesgo en simulación (peor escenario −0,41 % frente a −2,22 %); en
   ABIDES la diferencia es pequeña.
4. Los datos reales de ABIDES (25 %) dan el mejor resultado en mercados normales de ABIDES, pero no podemos separar
   su efecto del de los 60k pasos extra.

### 5.2 Comparación final con semillas de test

Estas semillas no se usaron para entrenar ni para elegir checkpoints. Simulación propia: 12 regímenes con semillas de test (5001–5005). ABIDES-NG en modo individual: 6 escenarios × semillas 3, 4 y 5. Comparamos el modelo entregado (`ft_abides25`), nuestro otro candidato fuerte (`v2_dueling_s11`) y el DQN starter. HOLD obtiene 0 en todo por definición.

#### Simulador propio (12 regímenes)

| Política | Retorno medio | Drawdown máx. medio | Peor escenario | Operaciones | Score |
|---|---|---|---|---|---|
| v2_dueling_s11 | 0,73 % | 0,39 % | −0,09 % | 204 | **0,51** |
| **ft_abides25 (entregado)** | **1,03 %** | 0,59 % | −1,49 % | 231 | 0,36 |
| momentum (regla) | 0,36 % | 0,98 % | −4,45 % | 194 | −1,24 |
| mean_reversion (regla) | −0,20 % | 1,14 % | −3,46 % | 127 | −1,64 |
| baseline (DQN starter) | −0,67 % | 2,47 % | −7,25 % | 128 | −3,72 |

#### Mercado de práctica oficial (3 regímenes)

| Política | Retorno medio | Drawdown máx. medio | Peor escenario | Operaciones | Score |
|---|---|---|---|---|---|
| baseline (DQN starter) | 1,81 % | 1,84 % | −0,14 % | 62 | 0,86 |
| v2_dueling_s11 | 0,94 % | 0,33 % | +0,04 % | 171 | 0,78 |
| **ft_abides25 (entregado)** | 0,49 % | 0,36 % | −0,34 % | 164 | 0,22 |

#### ABIDES-NG (6 escenarios × 3 semillas)

| Política | Media sin stress | maxDD sin stress | Peor escenario (PnL) | Inventario máx. medio |
|---|---|---|---|---|
| **ft_abides25 (entregado)** | **+$550** | **$885** | −$120 | **280** |
| v2_dueling_s11 | −$1.271 | $2.643 | −$6.176 | 331 |
| baseline (DQN starter) | −$8.179 | $14.031 | −$40.655 | 445 |

#### Lectura

1. **Frente al DQN starter**, `ft_abides25` mejora en casi todo: pierde menos y tiene mucho menos drawdown, tanto en simulación como en ABIDES. La excepción es el mercado de práctica, donde el starter rinde más porque está ajustado a esos 3 regímenes, cuya tendencia siempre es alcista.
2. **Entre nuestros dos candidatos no hay un ganador único.** En el simulador propio y en la práctica oficial, `v2_dueling_s11` tiene mejor score y menos cola de pérdida. En ABIDES-NG, `ft_abides25` es claramente mejor: gana en promedio, tiene un tercio del drawdown y menos inventario, mientras que `v2_dueling_s11` pierde en promedio por un mal día volátil (−$6.176).
3. **Las cifras de `stress_test` de ABIDES no son evidencia de habilidad** (por ejemplo, los +$47.715 de `ft_abides25` o los +$123.606 del starter). Con un máximo de 500 acciones, esos PnL superan el capital inicial por saltos de precio cuando un lado del libro se vacía. Por eso las columnas "sin stress" los excluyen.
4. **El peor escenario simulado de `ft_abides25` (−1,49 %) es el régimen `stress`**, donde el fine-tuning con datos de ABIDES aumenta la exposición, como ya habíamos visto en validación.
5. **Evidencia moderada, no definitiva.** En ABIDES solo hay 3 semillas por escenario, y las diferencias entre candidatos son del orden de la variación entre semillas.

### 5.3 Curvas de entrenamiento

`outputs/runs/<run>/training_curve.png`: recompensa por episodio (izquierda) y score, retorno y drawdown en validación cada 10k pasos (derecha). Datos crudos en `train_log.csv` y `val_log.json`.

## 6. Selección del modelo final

1. Durante cada corrida, el checkpoint se elige por el **score de validación** en semillas que nunca se usaron para entrenar: `score = retorno medio % − 0,5·drawdown máx. medio % + 0,25·retorno del peor escenario %`. Ese score premia la ganancia, castiga el riesgo y castiga depender de un solo régimen.
2. Entre corridas, los candidatos se comparan además **dentro de ABIDES-NG** (modo individual, 5 escenarios × 2 semillas). Así se evita elegir un modelo que solo funciona en nuestro simulador. En ABIDES se ignora el PnL y el drawdown de `stress_test` (ver limitaciones) y se mira el inventario máximo.
   *Nota:* la versión inicial del evaluador compartía semilla y precio entre escenarios, y con ella se eligieron los checkpoints de cada corrida. Lo corregimos después de la revisión, pero no reentrenamos.
3. El modelo elegido se evalúa **una sola vez** en las semillas de test (5001–5005, y ABIDES 3–5) para reportar números sin sesgo de selección.
4. Teníamos dos finalistas, `v2_dueling_s11` y `ft_abides25`. Nuestra **decisión final y modelo entregado** `ft_abides25` porque ABIDES-NG es el entorno más cercano al torneo y no al simulador con el que entrenamos. Ahí `ft_abides25` obtuvo mejor resultado medio, tres veces menos drawdown y menos inventario. Aceptamos a cambio un peor score en nuestro simulador y en el mercado de práctica, y un peor escenario más profundo en crisis.

## 7. Limitaciones y fallas observadas

- **Brecha simulador–ABIDES.** En ABIDES los retornos por minuto son unas 4 veces menores, el spread es casi nulo y el desbalance del libro varía unas 5 veces más que en el mercado de práctica. `stress_test` tiene saltos de varios por ciento por minuto con un spread de unos 90 pb. Un modelo entrenado solo en el mercado de práctica ve en ABIDES observaciones que nunca vio. Lo mitigamos con escenarios calibrados y con datos reales de ABIDES (`abides_collect.py`).
- **Sobre-operación** en mercados tranquilos: los modelos v1 operan en casi cada decisión, y el spread y el impacto se comen la poca ventaja que existe.
- **Asimetría de caja.** El mercado de práctica (y nuestro entorno, para ser consistentes) prohíbe comprar más allá de la caja, pero permite vender en corto hasta −500. A precios altos esto sesga cualquier política hacia posiciones cortas: incluso una política aleatoria gana en tendencias bajistas. Puede ser un artefacto del entorno y no una habilidad del agente.
- **Adaptador de ABIDES propio.** Nuestro `abides_eval.py` hace supuestos (decidir cada 60 s, profundidad = 5 niveles, cancelar las órdenes limitadas tras 5 decisiones) que el adaptador del profesor puede no compartir.
- **Pocas semillas en ABIDES**, porque cada día simulado tarda unos 40 s.
- **`stress_test` de ABIDES no sirve para comparar por PnL ni drawdown:** con como mucho 500 acciones aparecen PnL y drawdowns mayores que el capital inicial, por saltos del precio medio cuando un lado del libro se vacía, y varían en órdenes de magnitud entre semillas. Ahí solo comparamos exposición (inventario máximo).
- **El modelo entregado asume más riesgo en crisis:** En simulación, `ft_abides25` tiene su peor resultado en el régimen `stress` (−1,49 %, drawdown 2,0 %), y su drawdown medio (0,59 %) es mayor que el de `v2_dueling_s11` (0,39 %). En el mercado de práctica oficial rinde menos que el starter y que `v2_dueling_s11`. Su inventario máximo medio en ABIDES es 280 acciones, por debajo del límite de 500 pero todavía alto.
- **Simplificaciones del simulador propio** (detectadas en la revisión, no corregidas para no invalidar los modelos entrenados): una orden limitada nueva puede ejecutarse o perder un paso de vida en el mismo paso en que se coloca; al colocarla no se comprueba el límite de ±500 (si al ejecutarse lo rompería, se descarta); el impacto propio no alimenta el término de momentum.
- **Simplificaciones del adaptador de ABIDES:** las órdenes de mercado parcialmente ejecutadas no se pueden cancelar y siguen contando como pendientes; cuando un lado del libro está vacío el agente no decide en ese paso.
- **El fine-tuning con datos de ABIDES aumenta la exposición en crisis.** Las recompensas de ABIDES tienen colas enormes; las recortamos a ±50, pero el agente igual aprende posiciones más grandes.
- **El torneo puede desestabilizar el mercado:** con 5 agentes estratégicos, `volatile_day` semilla 2 produjo PnL de ±$900k.

## 8. Respuestas a las preguntas de discusión (sección 18)

**1. ¿Por qué reproducir un CSV histórico no equivale a interactuar con un exchange simulado?**
Un CSV es fijo: nuestras órdenes no cambian el precio ni la liquidez, y siempre se ejecutan al precio registrado. En un exchange simulado nuestras órdenes consumen liquidez (impacto), las limitadas pueden no ejecutarse, y otros agentes reaccionan a lo que hacemos. La transición de estado depende de nuestras acciones, así que es un problema de RL (y multi-agente), no de predicción supervisada. Además, un backtest sobre un CSV no puede generar contrafactuales: no sabemos qué habría pasado si hubiéramos actuado distinto.

**2. ¿Qué información del vector de observación ayuda a detectar baja liquidez?**
`spread_bps` (índice 4) se amplía. `bid_depth` y `ask_depth` (6–7) son bajos. `book_imbalance` (5) se vuelve extremo y errático. `volatility_10` (3) sube por saltos del precio medio con poco volumen. `open_orders` (12) se queda alto porque nuestras órdenes limitadas no se ejecutan.

**3. ¿Cuándo es preferible una orden limitada a una de mercado?**
Cuando el spread es amplio en relación con la ventaja esperada, cuando no hay urgencia (señal débil o de largo plazo), y cuando el libro es delgado, porque una orden de mercado grande paga impacto. La orden limitada captura el spread pero corre riesgo de no ejecutarse y de selección adversa (se ejecuta justo cuando el precio va en contra). La orden de mercado conviene cuando la señal es fuerte y de corto plazo, o para cerrar riesgo rápido (por ejemplo, en una crisis o al final del día).

**4. ¿Cómo puede una recompensa aparentemente rentable producir una política insegura?**
Si solo premia el PnL de corto plazo, el agente aprende a maximizar la exposición en la dirección que funcionó durante el entrenamiento. En nuestro caso la tendencia de práctica siempre es alcista, así que aprende "siempre comprado al límite". Gana en promedio pero sufre drawdowns enormes en crisis o tendencias bajistas. También puede aprender a sobre-operar si el modelo de ejecución es optimista (en práctica, las limitadas se ejecutan al 55 % a mejor precio). Nuestra comparación lo sugiere: con el reward del starter el agente acumula más inventario (468 frente a 210 acciones de máximo medio en ABIDES) y su drawdown en los mercados normales de ABIDES es unas 4–5 veces mayor, aunque en simulación obtiene más retorno.

**5. ¿Cómo distinguir generalización genuina de suerte en una semilla?**
Evaluar en muchas semillas y regímenes **no usados para entrenar ni para elegir el modelo**, y reportar media, desviación, peor caso y tasa de aciertos, no el mejor resultado. Separar validación (para elegir) de test (para reportar una vez). Comparar contra baselines con las mismas semillas. Repetir el entrenamiento con otra semilla (corremos v2 con semillas 7 y 11) para ver si la mejora se mantiene. Y, sobre todo, evaluar en un simulador distinto al de entrenamiento: ABIDES.

**6. ¿Por qué una política puede comportarse distinto cuando varios agentes estudiantes operan a la vez?**
Porque el entorno deja de ser estacionario: los otros agentes consumen liquidez, amplían el spread, mueven el precio con sus órdenes y pueden reaccionar a las mismas señales que nosotros (aglomeración). Si todos compran con la misma señal, el precio de entrada empeora y la señal pierde valor. Además, nuestras órdenes limitadas compiten en la cola con las de ellos. Una política entrenada contra agentes de fondo pasivos puede degradarse frente a agentes estratégicos. Lo observamos con `abides_eval.py --mode tournament`:
- con 5 políticas en el mismo exchange, en `volatile_day` semilla 2 el libro se desestabilizó y aparecieron PnL de ±$900k, algo que no ocurre con un solo agente;
- la regla de momentum, que en modo individual nunca operaba, empezó a operar porque los demás agentes movían el precio.

El escenario `crowded` de nuestro simulador modela de forma simple a otros seguidores de tendencia.

**7. ¿Qué compromiso hicimos entre ganancia, drawdown y actividad?**
Preferimos un modelo que no pierda mucho en ningún escenario antes que uno con el mejor retorno promedio. Eso se ve en dos niveles. Primero, el score de validación pondera el drawdown y el peor escenario, y por eso el DQN starter, con mejor retorno en práctica (1,81 %), queda muy por detrás en robustez (drawdown 1,84 %, y −7,25 % en su peor régimen simulado). Segundo, entre nuestros dos finalistas entregamos `ft_abides25`. Frente a `v2_dueling_s11` tiene más retorno en simulación (1,03 % contra 0,73 %) y mucho mejor resultado en ABIDES (+$550 contra −$1.271 sin stress, con drawdown de $885 contra $2.643). A cambio acepta más drawdown en simulación (0,59 % contra 0,39 %) y una pérdida mayor en crisis (−1,49 % contra −0,09 %). Elegimos ese compromiso porque ABIDES-NG se parece más al entorno del torneo. La actividad sigue siendo alta (unas 231 operaciones por episodio en simulación y 285 fills en ABIDES) y es la principal debilidad, porque el spread y el impacto se comen parte de la ventaja en mercados tranquilos.

**8. Si la política falla en una crisis, ¿cómo saber si el problema es la representación del estado, el reward, la distribución de entrenamiento o el algoritmo?**
- *Distribución de entrenamiento:* comparar el rendimiento en crisis con y sin escenarios de crisis en el entrenamiento (`--scenarios ...`). Si mejora al incluirlos, el problema era la cobertura.
- *Reward:* comparar presets de recompensa con lo demás fijo (lo hicimos: `starter` frente a `risk_aware`). Si con más penalización de drawdown mejora la crisis, el problema era el incentivo.
- *Representación del estado:* ablación de features (poner a cero una feature y medir la caída), o revisar si las observaciones en la crisis salen del rango visto en entrenamiento. Por ejemplo, la volatilidad normalizada se satura en 1,0 en `stress_test`, así que el agente no distingue "muy volátil" de "extremadamente volátil".
- *Algoritmo:* si con la misma distribución y el mismo reward la curva de validación es inestable entre semillas, o la pérdida diverge, el problema es de optimización (probar otro lr, Double DQN frente a DQN, comparar contra la ablación `v2_mlp_dqn`).
