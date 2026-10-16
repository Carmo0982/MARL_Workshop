# AGENTS.md — instrucciones para agentes de IA (Claude Code, Codex, Cursor, etc.)

Proyecto universitario (curso AIDA): entrenar un agente de RL que opera en bolsa y entregarlo como
`TEAM_XX.zip` con `model.pt` (TorchScript) + `metadata.json`. El trabajo de desarrollo **ya está hecho**.
Tu tarea es ayudar al compañero a **terminar**: pruebas finales, decisión del modelo, exportación e informe.

Contexto humano completo: `NOTAS_EQUIPO.md` (léelo antes de actuar). Informe: `INFORME_BORRADOR.md`.
Enunciado original: `MARL_Stock_Exchange_Workshop.docx`, `DELIVERABLE_SPECIFICATION.txt`.

## Reglas que no se rompen

1. **No modificar nada en `shared/` ni en `examples/`.** `shared/contract.py` es el contrato común con el profesor.
   Todo el código propio vive en `student_work/`.
2. El modelo entregado debe aceptar float32 `[batch, 16]` y devolver float32 `[batch, 9]`, ser TorchScript,
   correr en CPU y dar valores finitos. El torneo usa `argmax(output)`.
3. El ZIP final contiene **exactamente** `model.pt` y `metadata.json` en la raíz. Siempre generarlo con
   `shared.export_submission` y comprobarlo con `shared.validate_submission` hasta que imprima `VALID`.
4. **Semillas de test** (simulación 5001–5005, ABIDES 3–5): úsalas **una sola vez**, para reportar.
   No elijas ni ajustes modelos mirándolas. Para elegir, usa validación (1001–1003, ABIDES 1–2).
5. No borres `outputs/runs/`, `outputs/candidates/`, `outputs/eval/` ni `outputs/abides_eval/`: son la evidencia del informe.
6. No subas ni envíes nada (ZIP, informe) sin confirmación explícita del usuario.

## Entorno

- Ejecutar todo desde la raíz del proyecto con `.venv/bin/python -m <módulo>` (Windows: `.venv\Scripts\python.exe`).
- Si no existe `.venv`: `bash setup_mac.sh` (Mac/Linux) o `SETUP_WINDOWS_FIXED.bat` (Windows).
  **`abides-ng` no está en PyPI**: usar `requirements_fixed.txt`, nunca `requirements.txt`.
- Python 3.11–3.13 (con 3.14 torch puede no tener versión).
- Comprobación rápida: `.venv/bin/python -m shared.validate_submission submission/TEAM_XX.zip` → `VALID`.

## Tareas pendientes (en este orden)

1. **Verificar instalación:** correr los demos 1–5 y `shared.abides_smoke_test` (ver `run.sh`).
2. **Pruebas finales con semillas de test.** Comandos exactos en `NOTAS_EQUIPO.md` sección 8, paso 2. Comparar:
   - `outputs/model.pt` (= `outputs/candidates/v2_dueling_s11.pt`, elección provisional),
   - `outputs/candidates/ft_abides25.pt` (retador),
   - `outputs/baseline/model.pt` (DQN starter), `hold`, `mean_reversion`.
   El de ABIDES es lento (~40–90 s por simulación): lanza **un proceso por escenario en paralelo** y en segundo plano.
3. **Decidir el modelo** (la elección actual, `v2_dueling_s11`, es provisional) con este criterio, en orden de prioridad:
   - ABIDES sin `stress_test`: "maxDD excl. stress" bajo y "mean excl. stress" alta (columnas de `student_work.summarize_abides`);
   - exposición: "mean max abs(inv)" bajo, y que no llegue a ±500 de forma sistemática (incluido `stress_test`);
   - score de la simulación completa (`student_work.evaluate`), mirando también el peor escenario.
   El PnL y el drawdown de `stress_test` **no** se usan para decidir (ver trampas).
   Si cambia, `cp outputs/candidates/<elegido>.pt outputs/model.pt` y actualiza la sección 7 de `NOTAS_EQUIPO.md`.
4. **Exportar con el ID real** (pedírselo al usuario; no inventarlo):
   `.venv/bin/python -m shared.export_submission --team-id TEAM_NN --team-name "..." --members "A,B" --algorithm "Double Dueling DQN (n-step, domain randomization)" --model outputs/model.pt`
   y luego `shared.validate_submission submission/TEAM_NN.zip`. Borrar el provisional `submission/TEAM_XX.zip`.
5. **Informe:** completar los 3 bloques marcados `[COMPAÑERO]` en `INFORME_BORRADOR.md` con los números de test
   (copiar de `outputs/eval/FINAL_*/summary.txt` y `outputs/abides_eval/FINAL_test.md`). No inventar cifras:
   cada número debe salir de un archivo en `outputs/`.
6. *(Opcional, solo si el usuario lo pide)* mejoras listadas en `NOTAS_EQUIPO.md` sección 8, paso 6.

## Mapa del código (`student_work/`)

| Archivo | Para qué |
|---|---|
| `train.py` | Entrenador (Double/Dueling DQN, n-step, aleatorización, selección por validación). `--help` |
| `model.py` | `QNetwork` (starter, lo usan los demos: no borrar) y `DuelingQNetwork` |
| `envs.py` | `RobustMarketEnv`, 12 escenarios (`SCENARIOS`), `RewardConfig`/`REWARD_PRESETS`, semillas de validación/test |
| `evaluate.py` | Evaluación multi-escenario/semilla de `.pt` o baselines → `outputs/eval/<out>/` (precio fijo $100, semilla propia por escenario) |
| `baselines.py` | Reglas: hold, random, momentum, mean_reversion, buy_and_hold |
| `abides_eval.py` | Corre políticas dentro de ABIDES-NG (`--mode individual|tournament`) |
| `abides_collect.py` | Recolecta transiciones reales de ABIDES (.npz) para `train.py --abides-data` |
| `summarize_abides.py` | Tabla resumen de los CSV de ABIDES |
| `train_starter.py` | Entrenador original intacto (referencia) |

## Trampas conocidas

- **Hilos de torch:** con varios procesos en paralelo, todo se vuelve ~10× más lento si torch usa todos los núcleos.
  El código ya fija 1–2 hilos; si escribes scripts nuevos usa `torch.set_num_threads(1)`.
- **zsh no separa palabras** en `$VAR` sin comillas: usa `${=VAR}` o arrays al construir comandos en bucles.
- **Logs con buffer:** los `print` sin `flush=True` en procesos con `nohup` aparecen tarde; no asumas que el proceso falló.
- **`pkill -f` / `pgrep -f`** con un patrón que también aparece en tu propio comando de espera lo mata/detecta a él mismo;
  usa el truco `student_work[.]abides_eval`.
- **PnL y drawdown de `stress_test` en ABIDES no son comparables:** con ≤500 acciones aparecen valores mayores que el
  capital (saltos del precio medio con el libro vacío), y cambian en órdenes de magnitud entre semillas. Mirar solo la
  exposición. (ABIDES es determinista para una misma semilla y el mismo código.)
- **El entrenador nunca escribe en `outputs/model.pt`:** por defecto guarda en `outputs/candidates/<run-name>.pt`.
  Para entregar un modelo hay que copiarlo explícitamente.
- Los `max_drawdown` de los CSV antiguos de ABIDES (`outputs/abides_eval/v1`, `v2`, `tournament.csv`) se calcularon
  antes de una corrección y pueden estar algo subestimados; las evaluaciones nuevas ya están bien.
- `abides_eval.py` hace supuestos propios (decisión cada 60 s, profundidad = 5 niveles, cancelación de límites tras
  5 decisiones); el adaptador del profesor puede diferir. Mencionarlo como limitación, no como hecho.
- `DELIVERABLE_SPECIFICATION.txt` dice "validar con opción 5": es un error, la validación es la opción **V**.

## Definición de "terminado"

- [ ] `submission/TEAM_NN.zip` con el ID real, y `shared.validate_submission` imprime `VALID` para ese archivo exacto.
- [ ] `outputs/model.pt` es el modelo dentro del ZIP.
- [ ] Resultados con semillas de test guardados en `outputs/eval/FINAL_*` y `outputs/abides_eval/FINAL_test.*`.
- [ ] `INFORME_BORRADOR.md` sin marcas `[COMPAÑERO]` y con cifras que coinciden con esos archivos.
- [ ] `NOTAS_EQUIPO.md` sección 7 refleja el modelo finalmente entregado.
