# Optuna: как `trial.suggest_*` выбирает гиперпараметры (механизм сэмплеров)

**Автор:** Батырханов Батырхан
**Стек:** ML, Python

**Исследуемая библиотека:** [Optuna](https://github.com/optuna/optuna)
**Версия:** `v5.0.0` (тег [`v5.0.0`](https://github.com/optuna/optuna/tree/v5.0.0), commit `01ddd17ab9f36c3d2817aed9afe376e1cb047d35`)

**Исследуемый механизм:** путь от вызова `trial.suggest_float(...)` до конкретного значения.
Цепочка `Study → Trial → BaseSampler (relative / independent sampling) → BaseStorage`:
- стратегия `BaseSampler` и её реализации (TPE, GP, CMA-ES, Random и другие);
- `IntersectionSearchSpace`;
- fallback-композиция (TPE → RandomSampler);
- декоратор `PartialFixedSampler`;
- собственный сэмплер как точка расширения.

## Структура

| Файл | Что показывает |
|---|---|
| `demo/00_ask_tell_step_by_step.py` | Один trial вручную через `ask()/suggest/tell()`. Точки останова помечены `BREAKPOINT` |
| `demo/01_trace_sampling.py` | Трассировка **всех** вызовов сэмплера во время `study.optimize()` и места в коде Optuna, откуда они пришли |
| `demo/02_custom_sampler.py` | Собственный `LocalSearchSampler(BaseSampler)` вместо встроенного |
| `demo/03_compare_samplers.py` | Одна и та же задача с разными стратегиями (Random / TPE / CMA-ES / GP / Local) и график |
| `demo/tracing_sampler.py` | `TracingSampler`: декоратор над любым `BaseSampler`, логирует вызовы |
| `demo/local_search_sampler.py` | `LocalSearchSampler`: hill climbing вокруг лучшего trial, правило 1/5 в `after_trial` |
| `demo/common.py` | Целевые функции (`quadratic`, `rf` — RandomForest, `dynamic` — SVC/RF с условными параметрами) и фабрика сэмплеров |
| `docs/diagrams/` | UML-диаграммы (PlantUML-исходники и PNG) |
| `docs/cheatsheet.md` | Шпаргалка к защите |

## Запуск

Нужен Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# только для GPSampler (--sampler gp):
pip install torch --index-url https://download.pytorch.org/whl/cpu

python demo/00_ask_tell_step_by_step.py
python demo/01_trace_sampling.py                       # TPE, 15 trials
python demo/01_trace_sampling.py --sampler random
python demo/01_trace_sampling.py --sampler tpe-independent
python demo/01_trace_sampling.py --objective dynamic --n-trials 14 --no-caller
python demo/01_trace_sampling.py --enqueue x=2,y=-1,kind=b
python demo/01_trace_sampling.py --fix kind=b
python demo/01_trace_sampling.py --n-jobs 2 --n-trials 4
python demo/02_custom_sampler.py
python demo/03_compare_samplers.py                     # сохранит compare.png
```

Параметры `01`/`02`/`03`:
- `--objective {quadratic,rf,dynamic}`
- `--sampler {random,tpe,tpe-independent,tpe-group,cmaes,gp,qmc,local}`
- `--n-trials N`
- `--n-startup-trials N`
- `--seed N`

### Отладка (VS Code / PyCharm)

В `.vscode/launch.json` есть готовые конфигурации с `justMyCode: false`, чтобы отладчик заходил внутрь Optuna.
Ключевые точки останова в установленном пакете (`.venv/lib/python3.*/site-packages/optuna/`):

| Файл | Метод | Что происходит |
|---|---|---|
| `study/_optimize.py` | `_run_trial` | ask → objective → tell |
| `study/study.py` | `Study.ask` | создание trial в storage |
| `trial/_trial.py` | `Trial.__init__` | `sampler.before_trial` |
| `trial/_trial.py` | `Trial.relative_params` | `infer_relative_search_space` + `sample_relative` |
| `trial/_trial.py` | `Trial._suggest` | выбор: fixed / single / relative / independent |
| `samplers/_tpe/sampler.py` | `TPESampler.sample_independent`, `_sample` | |
| `study/_tell.py` | `_tell_with_warning` | `sampler.after_trial` + запись состояния |

## Использование ИИ

См. [AI_USAGE.md](AI_USAGE.md).
