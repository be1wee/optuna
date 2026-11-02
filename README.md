# Optuna: механизм сэмплеров

**Автор:** Бейшеев Абай. **Язык:** Python.

**Библиотека:** [Optuna 5.0.0](https://github.com/optuna/optuna/tree/v5.0.0).
Исследуем путь `Trial.suggest_float → BaseSampler → Storage` и замену алгоритма поиска.

Задача: минимизировать `(x - 2)² + (y + 1)²`. Минимум — 0 при `x=2, y=-1`.

## Запуск

Нужен Python 3.10+.

```powershell
python -m pip install -r requirements.txt
python demo/steps.py
python demo/run.py --sampler random
python demo/run.py --sampler tpe
python demo/custom.py
```

## Файлы

- `common.py` — формула оценки.
- `steps.py` — ручной цикл ask → objective → tell.
- `run.py` — 15 попыток с Random или TPE.
- `custom.py` — запуск собственного сэмплера.
- `local_sampler.py` — поиск вокруг лучших параметров.
- `tracing.py` — краткий вывод: попытка, способ выбора, параметры, результат.

`independent` — отдельный выбор параметров; `relative` — совместный.
Меньший результат лучше. Точное нахождение минимума не гарантируется.
