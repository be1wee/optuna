import random
from typing import Any

from optuna.distributions import BaseDistribution, FloatDistribution
from optuna.samplers import BaseSampler, RandomSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.trial import FrozenTrial, TrialState


class LocalSearchSampler(BaseSampler):
    def __init__(
        self, seed: int = 42, n_startup_trials: int = 3, radius: float = 0.1
    ) -> None:
        if n_startup_trials < 1 or radius <= 0:
            raise ValueError("Нужно n_startup_trials >= 1 и radius > 0")
        self._rng = random.Random(seed)
        self._random_sampler = RandomSampler(seed=seed)
        self._search_space = IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._radius = radius

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return self._search_space.calculate(study)

    def sample_relative(
        self, study: Study, trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        self._raise_error_if_multi_objective(study)
        completed_trials = study.get_trials(states=(TrialState.COMPLETE,))
        if len(completed_trials) < self._n_startup_trials:
            return {}

        parameters = {}
        for name, distribution in search_space.items():
            distribution = self._check_distribution(distribution)
            best_value = study.best_trial.params[name]
            deviation = self._radius * (distribution.high - distribution.low)
            candidate = self._rng.gauss(best_value, deviation)
            parameters[name] = max(distribution.low, min(distribution.high, candidate))
        return parameters

    def sample_independent(
        self, study: Study, trial: FrozenTrial,
        param_name: str, param_distribution: BaseDistribution,
    ) -> Any:
        self._raise_error_if_multi_objective(study)
        self._check_distribution(param_distribution)
        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    @staticmethod
    def _check_distribution(distribution: BaseDistribution) -> FloatDistribution:
        if (
            not isinstance(distribution, FloatDistribution)
            or distribution.log
            or distribution.step is not None
        ):
            raise ValueError("Поддерживаются только float без log и step")
        return distribution

    def reseed_rng(self) -> None:
        self._rng.seed()
        self._random_sampler.reseed_rng()
