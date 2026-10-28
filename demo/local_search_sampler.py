"""LocalSearchSampler — собственный сэмплер, реализующий интерфейс BaseSampler.

Идея (простой hill climbing):
  * первые n_startup_trials — случайный поиск (делегируем RandomSampler);
  * далее sample_relative берёт лучший trial и «шевелит» каждый его параметр
    гауссовым шумом с радиусом self._radius (в долях от ширины диапазона);
    категориальный параметр меняется на случайный с вероятностью categorical_flip_prob;
  * after_trial адаптирует радиус по «правилу 1/5»: улучшение -> расширяем,
    иначе — сужаем.
  * параметры вне пересечения пространств (динамические) -> sample_independent -> Random.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.samplers import BaseSampler, RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


class LocalSearchSampler(BaseSampler):
    def __init__(
        self,
        n_startup_trials: int = 10,
        radius: float = 0.2,
        categorical_flip_prob: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self._n_startup_trials = n_startup_trials
        self._radius = radius
        self._categorical_flip_prob = categorical_flip_prob
        self._rng = np.random.RandomState(seed)
        self._independent_sampler = RandomSampler(seed=seed)
        self._search_space = IntersectionSearchSpace()
        self._best_value = math.inf

    # 1) Какие параметры сэмплируем совместно
    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return {
            name: dist
            for name, dist in self._search_space.calculate(study).items()
            if not dist.single()
        }

    # 2) Совместное сэмплирование: окрестность лучшего trial
    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)
        if not search_space:
            return {}
        complete = study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True)
        if len(complete) < self._n_startup_trials:
            return {}  # -> все параметры пойдут в sample_independent (random)

        best = study.best_trial
        params = {}
        for name, dist in search_space.items():
            if name in best.params:
                params[name] = self._perturb(best.params[name], dist)
        return params

    # 3) Одиночный параметр вне relative-пространства
    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    # 4) Хук после завершения trial: адаптация радиуса
    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if state != TrialState.COMPLETE or values is None:
            return
        value = values[0]
        if study.direction == optuna.study.StudyDirection.MAXIMIZE:
            value = -value
        if value < self._best_value:
            self._best_value = value
            self._radius = min(self._radius * 1.5, 0.5)
        else:
            self._radius = max(self._radius * 0.9, 0.01)

    def reseed_rng(self) -> None:
        self._rng.seed()
        self._independent_sampler.reseed_rng()

    # ---- вспомогательное ----
    def _perturb(self, value: Any, dist: BaseDistribution) -> Any:
        if isinstance(dist, CategoricalDistribution):
            if self._rng.rand() < self._categorical_flip_prob:
                return dist.choices[self._rng.randint(len(dist.choices))]
            return value

        assert isinstance(dist, (FloatDistribution, IntDistribution))
        low, high, v = float(dist.low), float(dist.high), float(value)
        if dist.log:
            low, high, v = math.log(low), math.log(high), math.log(v)
        v = float(np.clip(v + self._rng.normal(0.0, self._radius * (high - low)), low, high))
        if dist.log:
            v = math.exp(v)
            v = min(max(v, dist.low), dist.high)
        if dist.step is not None:
            v = dist.low + round((v - dist.low) / dist.step) * dist.step
            v = min(max(v, dist.low), dist.high)
        return int(round(v)) if isinstance(dist, IntDistribution) else v
