"""Шаг за шагом: что происходит внутри одного trial (без study.optimize).

Удобно запускать под отладчиком: точки останова помечены «BREAKPOINT».
study.optimize() делает то же самое в цикле: optuna/study/_optimize.py::_run_trial.
"""

from __future__ import annotations

import optuna

from common import quiet_optuna
from tracing_sampler import TracingSampler


def main() -> None:
    quiet_optuna()

    # 1. Создаём Study. Внутри: storages.get_storage(None) -> InMemoryStorage,
    #    sampler сохраняется в study.sampler (по умолчанию был бы TPESampler()).
    sampler = TracingSampler(optuna.samplers.TPESampler(seed=0, n_startup_trials=3))
    study = optuna.create_study(direction="minimize", sampler=sampler)

    for _ in range(5):
        # 2. ask(): storage.create_new_trial(...) -> Trial(study, trial_id)
        #    Trial.__init__ вызывает sampler.before_trial(...)
        trial = study.ask()  # BREAKPOINT 1: optuna/study/study.py  Study.ask

        # 3. suggest_*: Trial._suggest -> (relative_params | sample_independent)
        #    -> storage.set_trial_param(...)
        x = trial.suggest_float("x", -10, 10)  # BREAKPOINT 2: optuna/trial/_trial.py Trial._suggest
        y = trial.suggest_float("y", -10, 10)
        # Повторный suggest с тем же именем НЕ вызывает сэмплер: значение берётся из trial.params
        x_again = trial.suggest_float("x", -10, 10)
        assert x == x_again

        value = (x - 2) ** 2 + (y + 1) ** 2

        # 4. tell(): _tell_with_warning -> sampler.after_trial -> storage.set_trial_state_values
        study.tell(trial, value)  # BREAKPOINT 3: optuna/study/_tell.py _tell_with_warning
        print(f"  -> value={value:.4f}")

    print("\nЛучший trial:", study.best_trial.number, study.best_params, f"{study.best_value:.4f}")
    print("Storage:", type(study._storage).__name__)


if __name__ == "__main__":
    main()
