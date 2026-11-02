from optuna.samplers import BaseSampler


class TracingSampler(BaseSampler):
    def __init__(self, base):
        self._base = base

    def before_trial(self, study, trial):
        self._mode = "independent"
        self._base.before_trial(study, trial)

    def infer_relative_search_space(self, study, trial):
        return self._base.infer_relative_search_space(study, trial)

    def sample_relative(self, study, trial, search_space):
        parameters = self._base.sample_relative(study, trial, search_space)
        if parameters:
            self._mode = "relative"
        return parameters

    def sample_independent(self, study, trial, param_name, param_distribution):
        value = self._base.sample_independent(study, trial, param_name, param_distribution)
        return value

    def after_trial(self, study, trial, state, values):
        if values is not None:
            parameters = " ".join(f"{name}={value:.3f}" for name, value in trial.params.items())
            print(f"{trial.number + 1:2} | {self._mode:11} | {parameters} | f={values[0]:.3f}")
        else:
            print(f"{trial.number + 1:2} | {state.name}")
        self._base.after_trial(study, trial, state, values)

    def reseed_rng(self):
        self._base.reseed_rng()
