# NOTAS DEL EQUIPO — MARL Stock Exchange

Documento de traspaso: qué está hecho, cómo correrlo, qué encontramos y **qué falta**.
Léelo completo antes de tocar el código. El borrador del informe está en `INFORME_BORRADOR.md`.

---

## 0. Estado actual

| Parte | Estado |
|---|---|
| Instalación en Mac (y arreglo para Windows) | ✅ hecho: `setup_mac.sh`, `run.sh`, `SETUP_WINDOWS_FIXED.bat` |
| Escenarios extra + recompensa con riesgo | ✅ `student_work/envs.py` |
| Evaluador multi-escenario / multi-semilla + baselines | ✅ `student_work/evaluate.py`, `student_work/baselines.py` |
| Double + Dueling DQN, n-step, checkpoints, selección por validación | ✅ `student_work/train.py`, `student_work/model.py` |
| Adaptador para correr el modelo **dentro de ABIDES-NG** (individual y torneo) | ✅ `student_work/abides_eval.py` |
| Recolección de datos reales de ABIDES para entrenar | ✅ `student_work/abides_collect.py` |
| Modelo candidato entrenado y copiado a `outputs/model.pt` | ✅ ver sección 7 |
| ZIP provisional validado (`submission/TEAM_XX.zip`) | ✅, **re-exportar con el ID real del equipo** |
| Pruebas finales con semillas de test + números finales del informe | ⏳ **compañero** (sección 8) |
| Exportar con el ID real, validar y subir | ⏳ **compañero** |

**Reglas que respetamos:** no tocamos nada de `shared/` (es el contrato común con el profesor). Todo nuestro código está en `student_work/`. El modelo sigue siendo `[batch,16] → [batch,9]` en TorchScript.

---

## 1. Instalación

> ⚠️ **`abides-ng` NO está en PyPI.** El `requirements.txt` y el `SETUP_ONCE.bat` originales fallan con
> `No matching distribution found for abides-ng`. Usa `requirements_fixed.txt`, que lo instala desde GitHub sin necesidad de tener git.

**Mac / Linux** (Python 3.11–3.13; con 3.14 PyTorch puede no tener versión):
```bash
bash setup_mac.sh      # una vez
bash run.sh            # menú equivalente a START_HERE.bat + herramientas nuevas
```

**Windows:** ejecutar `SETUP_WINDOWS_FIXED.bat` (en lugar de `SETUP_ONCE.bat`) y luego `START_HERE.bat` como siempre.
Para las herramientas nuevas, usar los mismos comandos de abajo cambiando `.venv/bin/python` por `.venv\Scripts\python.exe`.

Otra nota: en `DELIVERABLE_SPECIFICATION.txt` dice "validar con la opción 5", pero en el menú la validación es la opción **V** (la 5 es el demo 5).

---

## 2. Archivos nuevos

```
student_work/
├── model.py            QNetwork (starter, lo usan los demos) + DuelingQNetwork (el nuestro)
├── train.py            Entrenador mejorado (Double/Dueling/n-step/aleatorización/selección)
├── train_starter.py    Copia intacta del entrenador original (para reproducir el baseline)
├── envs.py             RobustMarketEnv: 12 regímenes, órdenes limitadas que quedan en el libro, impacto, RewardConfig
├── baselines.py        Políticas por reglas: hold, random, momentum, mean_reversion, buy_and_hold
├── evaluate.py         Evaluación multi-escenario × multi-semilla → tabla, CSV, gráfica
├── abides_eval.py      Corre cualquier política dentro de ABIDES-NG (modo individual o torneo)
├── abides_collect.py   Junta transiciones reales de ABIDES para entrenar con ellas
└── summarize_abides.py Resume los CSV de ABIDES en una tabla
outputs/
├── model.pt            ← MODELO ACTUAL PARA ENTREGAR
├── baseline/model.pt   DQN starter (5.000 pasos, semilla 7): referencia
├── candidates/*.pt     Todos los modelos candidatos
├── runs/<corrida>/     config.json, train_log.csv, val_log.json, training_curve.png, checkpoints
├── eval/<nombre>/      summary.txt, episodes.csv, pnl_by_scenario.png
├── abides_eval/        CSV de las evaluaciones en ABIDES
└── abides_data/        Transiciones recolectadas en ABIDES (.npz)
```

---

## 3. Comandos principales

Todos se corren desde la carpeta del proyecto. En el menú `bash run.sh` están los más usados.

```bash
# Entrenar (unos 10-15 min con la CPU libre). Guarda el mejor checkpoint en --out
# (por defecto outputs/candidates/<run-name>.pt; nunca sobrescribe outputs/model.pt)
.venv/bin/python -m student_work.train --steps 200000 --seed 7 --run-name mi_corrida --out outputs/candidates/mi_corrida.pt

# Comparar modelos y baselines en los 12 escenarios (semillas de validación)
.venv/bin/python -m student_work.evaluate --policies outputs/model.pt outputs/baseline/model.pt hold momentum mean_reversion --suite full --seeds val --out outputs/eval/comparacion

# Lo mismo, pero en el mercado de práctica oficial (equivale a la opción 8 del menú)
.venv/bin/python -m student_work.evaluate --policies outputs/model.pt outputs/baseline/model.pt --suite practice --seeds val

# Evaluar dentro de ABIDES-NG (unos 40 s por día simulado y política)
.venv/bin/python -m student_work.abides_eval --policies outputs/model.pt outputs/baseline/model.pt hold --scenarios stable_day trending_day volatile_day low_liquidity stress_test --seeds 1 2 --out outputs/abides_eval/mi_eval.csv
.venv/bin/python -m student_work.summarize_abides outputs/abides_eval/mi_eval.csv

# Torneo: todos en el MISMO exchange (la parte multi-agente)
.venv/bin/python -m student_work.abides_eval --mode tournament --policies outputs/model.pt outputs/baseline/model.pt momentum mean_reversion --scenarios stable_day volatile_day --seeds 1

# Datos reales de ABIDES y fine-tuning con ellos
.venv/bin/python -m student_work.abides_collect --policy outputs/model.pt --seeds 11 12 13 14 --out outputs/abides_data/round1.npz
# (comando exacto con el que se creó ft_abides25; para ft_abides50 cambiar --abides-frac 0.5)
.venv/bin/python -m student_work.train --steps 60000 --init outputs/candidates/v2_dueling_s7.pt --eps-start 0.2 --eps-frac 0.3 --lr 2.5e-4 --eval-every 6000 --abides-data outputs/abides_data/round1.npz --abides-frac 0.25 --run-name finetune --out outputs/candidates/finetune.pt

# Exportar y validar (como las opciones 9 y V del menú)
.venv/bin/python -m shared.export_submission --team-id TEAM_XX --team-name "Nombre" --members "Persona A,Persona B" --algorithm "Double Dueling DQN (n-step)" --model outputs/model.pt
.venv/bin/python -m shared.validate_submission submission/TEAM_XX.zip
```

Opciones útiles de `train.py` (`--help` las muestra todas):
- `--reward starter|risk_aware|pnl_only|strong_risk`
- `--scenarios ...`
- `--no-randomize`
- `--arch mlp|dueling`
- `--no-double`
- `--n-step`
- `--abides-weight`
- `--practice-frac`
- `--limit-model instant|resting`

---

## 4. Lo que encontramos (importante para entender las decisiones)

1. **El mercado de práctica es muy distinto de ABIDES-NG.** Medimos las observaciones que ve nuestro agente en ABIDES (decidiendo cada 60 s) frente a las del mercado de práctica:

   | | std return_1 | std return_20 | spread (norm.) | std book_imbalance |
   |---|---|---|---|---|
   | ABIDES stable_day | 0,003 | 0,008 | 0,002 | **0,36** |
   | ABIDES volatile_day | 0,017 | 0,051 | 0,008 | 0,35 |
   | ABIDES stress_test | **0,60** | **0,83** | **0,44** | 0,45 |
   | Práctica stable | 0,014 | 0,064 | 0,020 | 0,07 |
   | Práctica volatile | 0,094 | 0,323 | 0,075 | 0,12 |

   En ABIDES, los mercados normales se mueven unas 4 veces menos, el spread es casi nulo y el desbalance del libro varía mucho más. `stress_test` tiene saltos enormes. Por eso agregamos 4 escenarios `abides_*` calibrados para reproducir estas estadísticas.
2. **La regla de momentum del demo 2 nunca opera en ABIDES:** su umbral es demasiado grande para movimientos tan pequeños.
3. **Las órdenes limitadas de práctica son irreales:** se ejecutan el 55 % de las veces al instante y a mejor precio que una orden de mercado, y `open_orders` siempre vale 0. En nuestro entorno quedan en el libro con un tiempo de vida.
4. **La tendencia de práctica siempre es alcista:** un agente puede aprender "siempre comprado". Agregamos `trend_down` y `stress`.
5. **Asimetría de caja:** está prohibido comprar más allá de la caja, pero se puede vender en corto hasta −500. A precios altos esto favorece las posiciones cortas; hasta la política aleatoria gana en tendencias bajistas. Tenerlo en cuenta al interpretar resultados.
6. **Rendimiento:** con varios procesos en paralelo hay que limitar torch a 1–2 hilos. Si no, todo se vuelve unas 10 veces más lento (ya está hecho en el código).

**Supuestos de nuestro adaptador de ABIDES** (el del profesor puede diferir):
- decide cada 60 s;
- profundidad = acciones en los 5 mejores niveles de cada lado;
- las órdenes limitadas se cancelan tras 5 decisiones;
- se aplica el límite de ±500 acciones;
- no se aplica la restricción de caja.

---

## 5. Mejoras implementadas (resumen para el informe)

| Dirección (sección 11) | Qué hicimos |
|---|---|
| Modelo | Dueling + LayerNorm, 128 unidades ocultas |
| Aprendizaje | Double DQN, n-step = 3, actualización suave de la red objetivo, lr con decaimiento, replay de 200k |
| Recompensa | Penalizaciones por inventario, pérdidas por paso (las pérdidas pesan 1,5×), rotación e inventario final |
| Regímenes de entrenamiento | 12 regímenes, aleatorización de parámetros, precio y duración; 4 regímenes calibrados con ABIDES |
| Exploración | ε lineal 1,0 → 0,02 en el 40 % del entrenamiento |
| Robustez / selección | Checkpoint elegido por score en semillas de validación no vistas; comparación final dentro de ABIDES |
| Sim-to-real | Datos reales de ABIDES mezclados en el replay (`abides_collect.py` + `--abides-data`) |

---

## 6. Resultados

Todos los números: semillas de **validación** (las de test todavía no se usaron, ver sección 8). PnL en dólares, capital inicial de $100.000.

### 6.1 Candidatos entrenados

| Candidato | Qué es | Pasos |
|---|---|---|
| `ddqn_risk_aware` | v1: Double + Dueling + n-step, recompensa con riesgo, 8 regímenes (sin los `abides_*` y con una versión anterior, más extrema, de `stress`) | 150k |
| `ddqn_starter_reward` | v1, igual pero con la recompensa original → **ablación de recompensa** | 150k |
| `v2_dueling_s7` / `v2_dueling_s11` | v2: igual que v1 + 4 regímenes calibrados con ABIDES (peso 2×). Semillas 7 y 11 → **variación entre semillas** | 200k |
| `v2_mlp_dqn_s7` | v2 con el MLP del starter, DQN simple y n-step = 1 → **ablación de algoritmo y arquitectura** | 200k |
| `ft_abides25` / `ft_abides50` | fine-tuning de `v2_dueling_s7` con 6.581 transiciones reales de ABIDES (25 % / 50 % de cada batch) | +60k |
| `baseline/model` | DQN starter original (5.000 pasos) | 5k |

> Nota: durante el entrenamiento, el mejor checkpoint de cada corrida se eligió con la versión antigua del evaluador
> (ver 6.2). No se reentrenó; las comparaciones de abajo sí usan el evaluador corregido.

### 6.2 Mercados simulados: 12 regímenes × 3 semillas (`outputs/eval/v3_full_fixed/`)

> **Corrección tras la revisión:** la primera versión del evaluador usaba la misma semilla en los 12 escenarios y un
> precio inicial aleatorio entre $50 y $1.500. Como el límite es en acciones y el retorno en dólares, una sola semilla
> de precio alto (1002, unos $1.178) concentraba el 80 % del PnL. Ahora cada escenario tiene su propia semilla y el
> precio de evaluación es fijo en $100. Las tablas antiguas (`outputs/eval/v1_full`, `v2_full`) quedan solo como historial.

| Política | Retorno medio | Drawdown máx. medio | Peor escenario | Operaciones | **Score** |
|---|---|---|---|---|---|
| ft_abides50 | 0,89 % | 0,43 % | −0,23 % | 219 | **0,62** |
| v2_dueling_s7 | 0,87 % | 0,42 % | −0,41 % | 217 | **0,56** |
| ddqn_starter_reward | 1,14 % | 1,24 % | −0,44 % | 178 | 0,41 |
| **v2_dueling_s11 (actual)** | 0,67 % | **0,37 %** | −0,34 % | 188 | 0,40 |
| ft_abides25 | 0,91 % | 0,63 % | −1,18 % | 218 | 0,30 |
| hold | 0 | 0 | 0 | 0 | 0,00 |
| v2_mlp_dqn_s7 | 1,07 % | 1,03 % | −2,22 % | 237 | 0,00 |
| ddqn_risk_aware | 0,34 % | 0,83 % | −0,57 % | 223 | −0,21 |
| mean_reversion (regla) | 0,07 % | 0,80 % | −1,90 % | 125 | −0,80 |
| momentum (regla del demo) | 0,37 % | 0,88 % | −4,04 % | 187 | −1,07 |
| baseline (DQN starter) | −0,75 % | 2,28 % | −4,38 % | 125 | **−2,99** |

Lectura: todos los modelos RL superan claramente al DQN starter, pero entre ellos las diferencias son pequeñas
(del orden de la variación entre semillas: `v2_dueling_s7` 0,56 frente a `v2_dueling_s11` 0,40). En simulación, la
recompensa con riesgo **no** sale mejor que la original (0,41 frente a −0,21 en v1); su ventaja aparece en ABIDES (6.3).
Detalle por escenario en `outputs/eval/v3_full_fixed/summary.txt` y `pnl_by_scenario.png`.

En el **mercado de práctica oficial** (3 regímenes, `outputs/eval/v2_practice/`, no afectado por la corrección), el DQN
starter tiene el mayor retorno medio (3,1 %, contra 0,5 % de `v2_dueling_s11`), pero también el mayor drawdown
(1,66 % contra 0,45 %). Está ajustado justo a esos 3 regímenes y no generaliza.

### 6.3 ABIDES-NG, modo individual: 5 escenarios × 2 semillas (`outputs/abides_eval/v2/summary.md`)

| Política | low_liq | stable | stress | trending | volatile | **Media sin stress** | **maxDD sin stress** | max abs(inv) medio |
|---|---|---|---|---|---|---|---|---|
| ft_abides25 | −38 | −179 | 56.974 | 280 | 873 | **234** | 355 | 237 |
| v2_dueling_s11 (actual) | −70 | −89 | 4.688 | −112 | 361 | 22 | 819 | 367 |
| v2_dueling_s7 | −329 | −104 | 3.902 | 72 | −911 | −318 | 1.599 | 143 |
| ddqn_risk_aware | −671 | −194 | 5.717 | 177 | −232 | −230 | 478 | 210 |
| ddqn_starter_reward | −360 | 37 | 241.302 | 709 | −1.765 | −345 | 2.220 | 468 |
| ft_abides50 | 155 | −244 | −42.810 | −1.112 | −760 | −490 | 910 | 404 |
| v2_mlp_dqn_s7 | 718 | 100 | 29.888 | −414 | −3.366 | −741 | 1.601 | 481 |
| baseline (starter) | −1.173 | −407 | −3.776 | −787 | −1.377 | **−936** | 2.286 | 496 |
| hold | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mean_reversion (regla) | 100 | 0 | 20.208 | 117 | 1.017 | 309 | **294** | 110 |

Cómo leer esta tabla:
- **No comparar por PnL ni drawdown en `stress_test`.** Con como mucho 500 acciones aparecen PnL y drawdowns mayores
  que todo el capital, lo que implica saltos absurdos del precio medio cuando un lado del libro se vacía. ABIDES es
  determinista para una semilla dada, pero de una semilla a otra estos números cambian en órdenes de magnitud.
  En stress, mirar solo la exposición (inventario máximo, si llega al límite de ±500).
- *Corrección:* antes atribuimos al ruido que `ddqn_starter_reward` diera −$177.268 en la evaluación v1 y +$241.302 en
  la v2. Era incorrecto: entre v1 y v2 corregimos el adaptador (en v1 no contaba las órdenes pendientes y ese modelo
  llegó a 560 acciones, por encima del límite).
- Los `max_drawdown` de estos CSV se calcularon antes de corregir `abides_eval.py` (no incluían el capital inicial ni
  el cierre), así que pueden estar algo subestimados. Las evaluaciones nuevas ya usan el cálculo corregido.
- En los mercados normales las ventajas son pequeñas. El starter pierde en los 5 escenarios. La regla `mean_reversion`
  funciona bien porque el precio fundamental de ABIDES revierte a la media.
- **Evidencia del revisor externo (semillas extra 7 y 8, no reproducida por nosotros):** `v2_dueling_s11` llegó a
  inventarios de 480–499 y perdió −$21.310 en `volatile_day` (semilla 7), mientras que `ft_abides25` tuvo +$602 de media
  sin stress con inventario máx. ≤ 287 en mercados normales. Ambos llegaron al límite en `stress_test`.

### 6.4 ABIDES-NG, modo torneo (`outputs/abides_eval/tournament_summary.md`)

Cinco políticas en el **mismo exchange** (stable, volatile y trending × 2 semillas):

| Política | stable | trending | volatile s1 | volatile s2 |
|---|---|---|---|---|
| v2_dueling_s11 | −181 | 404 | 1.294 | 645.054 |
| ft_abides25 | −35 | 871 | 484 | 909.890 |
| baseline (starter) | −827 | −1.341 | −238 | −215.899 |
| mean_reversion | 0 | 162 | 1.258 | 173.655 |
| momentum | −206 | −515 | −1.274 | −149.169 |

**Efecto multi-agente observado:** con varios agentes estratégicos operando a la vez, en `volatile_day` semilla 2 el libro se desestabiliza y aparecen PnL de cientos de miles de dólares. Eso no pasa con un solo agente. Además, la regla de momentum, que en modo individual nunca operaba, **sí opera en el torneo**, porque los otros agentes mueven más el precio. Este es material directo para la pregunta 6 del informe.

---

## 7. Modelo actual (PROVISIONAL): `outputs/model.pt` = `outputs/candidates/v2_dueling_s11.pt`

- **La elección final está abierta** y le toca al compañero, con las semillas de test (sección 8). `v2_dueling_s11`
  se eligió antes de la revisión, con el evaluador antiguo y con un criterio que daba peso al drawdown en
  `stress_test` (que resultó no ser fiable).
- Con los datos corregidos, los dos candidatos fuertes son:
  - **`v2_dueling_s11`:** el menor drawdown en simulación (0,37 %), pero en ABIDES llega a inventarios altos
    (367 de media; 480–499 en las semillas extra del revisor).
  - **`ft_abides25`:** el mejor en mercados normales de ABIDES (+234 de media, drawdown $355, inventario 237) y en el
    torneo, pero el peor "peor escenario" en simulación (−1,18 %).
- `submission/TEAM_XX.zip` contiene `v2_dueling_s11` y **pasa la validación**; hay que re-exportarlo con el modelo final
  y el ID real del equipo.
- Opción 8 del menú (semillas 101–103) con `v2_dueling_s11`: stable +$28, trending +$964, volatile +$17 (el starter: −$1.450, +$2.688, +$1.856).

---

## 8. Lo que falta (para ti, compañero)

1. **Instalar y verificar** (unos 10 min): `bash setup_mac.sh` o `SETUP_WINDOWS_FIXED.bat`, y después los demos 1–6 del menú.
2. **Pruebas finales con las semillas de TEST** (se usan una sola vez, al final; no elegir el modelo mirándolas):
   ```bash
   .venv/bin/python -m student_work.evaluate --policies outputs/model.pt outputs/candidates/ft_abides25.pt outputs/baseline/model.pt hold momentum mean_reversion --suite full --seeds test --out outputs/eval/FINAL_test
   .venv/bin/python -m student_work.evaluate --policies outputs/model.pt outputs/candidates/ft_abides25.pt outputs/baseline/model.pt hold --suite practice --seeds test --out outputs/eval/FINAL_practice
   .venv/bin/python -m student_work.abides_eval --policies outputs/model.pt outputs/candidates/ft_abides25.pt outputs/baseline/model.pt hold mean_reversion --scenarios stable_day trending_day volatile_day low_liquidity stress_test rmsc04 --seeds 3 4 5 --out outputs/abides_eval/FINAL_test.csv
   .venv/bin/python -m student_work.summarize_abides outputs/abides_eval/FINAL_test.csv --out outputs/abides_eval/FINAL_test.md
   ```
   El de ABIDES tarda unos 30–40 min. Se puede partir por escenario en varias terminales, como hicimos nosotros.
3. **Decidir** entre `v2_dueling_s11` (actual) y `ft_abides25` con esos resultados, usando el criterio de `AGENTS.md` (tarea 3). Si cambias, haz `cp outputs/candidates/ft_abides25.pt outputs/model.pt`.
4. **Exportar con el ID real** (opción 9 del menú o el comando de la sección 3), **validar el ZIP exacto** (opción V) y subirlo.
5. **Informe:** completar las partes marcadas **[COMPAÑERO]** en `INFORME_BORRADOR.md` con los números de test, y revisar las respuestas de la sección 8.
6. **Mejoras opcionales**, si hay tiempo (todas encajan con el código actual):
   - Más datos de ABIDES. Hacer una segunda ronda de `abides_collect.py` con el modelo actual y otras semillas (21–24), y fine-tunear con `--abides-data outputs/abides_data/round1.npz outputs/abides_data/round2.npz`.
   - Penalizar más el inventario **solo en crisis** (preset `strong_risk`) para que `ft_abides25` no acumule posiciones en `stress_test`.
   - Aprovechar que ABIDES revierte a la media: subir el peso de `mean_revert` / `abides_calm` en el entrenamiento (`--scenarios` o editar `weights`).
   - Más semillas de entrenamiento (p. ej. 3) y elegir la mejor por validación. Eso además da evidencia de varianza entre semillas.
