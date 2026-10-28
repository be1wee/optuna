"""TracingSampler — обёртка (Decorator/Proxy) над любым BaseSampler.

Не меняет поведение оборачиваемого сэмплера, а только печатает каждый вызов
метода интерфейса BaseSampler и место в коде Optuna, откуда он пришёл.
Это возможно именно потому, что Study/Trial зависят только от абстракции BaseSampler.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState

_OPTUNA_DIR = os.path.dirname(optuna.__file__)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}={_fmt(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _fmt_dist(d: BaseDistribution) -> str:
    args = ", ".join(f"{k}={_fmt(v)}" for k, v in d._asdict().items())
    return f"{type(d).__name__}({args})"


def _optuna_caller() -> str:
    """Первый кадр стека внутри пакета optuna, который вызвал сэмплер."""
    frame = sys._getframe(2)
    while frame is not None:
        path = frame.f_code.co_filename
        if path.startswith(_OPTUNA_DIR):
            rel = os.path.relpath(path, os.path.dirname(_OPTUNA_DIR))
            return f"{rel}:{frame.f_lineno} {frame.f_code.co_name}()"
        frame = frame.f_back
    return "<user code>"


class TracingSampler(BaseSampler):
    def __init__(self, base: BaseSampler, show_caller: bool = True) -> None:
        self._base = base
        self._show_caller = show_caller

    def __str__(self) -> str:
        return f"Tracing({self._base})"

    def _log(self, trial: FrozenTrial | None, msg: str) -> None:
        prefix = f"  [T{trial.number}]" if trial is not None else "  "
        thread = threading.current_thread().name
        thread = "" if thread == "MainThread" else f" <{thread}>"
        caller = f"\n        ↳ вызвано из {_optuna_caller()}" if self._show_caller else ""
        print(f"{prefix}{thread} {msg}{caller}")

    # ---- жизненный цикл trial ----
    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        print(f"\n=== Trial #{trial.number} (sampler={self._base}) ===")
        self._log(trial, "before_trial()")
        self._base.before_trial(study, trial)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        space = self._base.infer_relative_search_space(study, trial)
        names = ", ".join(space) if space else "∅"
        self._log(trial, f"infer_relative_search_space() -> {{{names}}}")
        return space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        params = self._base.sample_relative(study, trial, search_space)
        self._log(trial, f"sample_relative() -> {_fmt(params) if params else '{} (нет данных/пусто)'}")
        return params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        value = self._base.sample_independent(study, trial, param_name, param_distribution)
        self._log(
            trial,
            f"sample_independent('{param_name}', {_fmt_dist(param_distribution)}) -> {_fmt(value)}",
        )
        return value

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Any,
    ) -> None:
        self._log(trial, f"after_trial(state={state.name}, values={values}) params={_fmt(trial.params)}")
        self._base.after_trial(study, trial, state, values)

    def reseed_rng(self) -> None:
        print(f"  reseed_rng() <{threading.current_thread().name}>")
        self._base.reseed_rng()
