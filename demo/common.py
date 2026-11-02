import optuna


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    return (x - 2) ** 2 + (y + 1) ** 2


def quiet_optuna() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
