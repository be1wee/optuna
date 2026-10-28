"""Целевые функции и вспомогательный код, общие для всех демо-скриптов."""

from __future__ import annotations

import argparse
import logging

import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        _DATA = load_breast_cancer(return_X_y=True)
    return _DATA


def objective_quadratic(trial: optuna.Trial) -> float:
    """Быстрая аналитическая функция: минимум 0 в точке x=2, y=-1, kind="b"."""
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    kind = trial.suggest_categorical("kind", ["a", "b", "c"])
    penalty = {"a": 1.0, "b": 0.0, "c": 2.0}[kind]
    return (x - 2) ** 2 + (y + 1) ** 2 + penalty


def objective_rf(trial: optuna.Trial) -> float:
    """Подбор гиперпараметров RandomForest (статическое пространство поиска)."""
    X, y = _data()
    clf = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 10, 200, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 16),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        random_state=0,
        n_jobs=1,
    )
    # Минимизируем ошибку = 1 - accuracy.
    return 1.0 - cross_val_score(clf, X, y, cv=3).mean()


def objective_dynamic(trial: optuna.Trial) -> float:
    """Define-by-run: набор параметров зависит от выбранного классификатора.

    Параметры svc_* и rf_* не входят в *пересечение* пространств поиска всех trial'ов,
    поэтому TPESampler(multivariate=True) сэмплирует их через sample_independent.
    """
    X, y = _data()
    name = trial.suggest_categorical("classifier", ["svc", "rf"])
    if name == "svc":
        clf = make_pipeline(
            StandardScaler(),
            SVC(
                C=trial.suggest_float("svc_C", 1e-3, 1e3, log=True),
                gamma=trial.suggest_float("svc_gamma", 1e-4, 1e1, log=True),
            ),
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 10, 100, log=True),
            max_depth=trial.suggest_int("rf_max_depth", 2, 16),
            random_state=0,
        )
    return 1.0 - cross_val_score(clf, X, y, cv=3).mean()


OBJECTIVES = {
    "quadratic": objective_quadratic,
    "rf": objective_rf,
    "dynamic": objective_dynamic,
}


def make_sampler(name: str, seed: int, n_startup_trials: int = 10) -> optuna.samplers.BaseSampler:
    """Фабрика сэмплеров: именно здесь меняется *стратегия* подбора параметров."""
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    if name == "tpe-independent":
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=n_startup_trials, multivariate=False
        )
    if name == "tpe-group":
        return optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=n_startup_trials, multivariate=True, group=True
        )
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed, n_startup_trials=n_startup_trials)
    if name == "gp":
        return optuna.samplers.GPSampler(seed=seed, n_startup_trials=n_startup_trials)
    if name == "qmc":
        return optuna.samplers.QMCSampler(seed=seed)
    if name == "local":
        from local_search_sampler import LocalSearchSampler

        return LocalSearchSampler(seed=seed, n_startup_trials=n_startup_trials)
    raise ValueError(f"unknown sampler: {name}")


SAMPLER_NAMES = ["random", "tpe", "tpe-independent", "tpe-group", "cmaes", "gp", "qmc", "local"]


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--objective", choices=OBJECTIVES, default="quadratic")
    p.add_argument("--sampler", choices=SAMPLER_NAMES, default="tpe")
    p.add_argument("--n-trials", type=int, default=15)
    p.add_argument("--n-startup-trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p


def quiet_optuna() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)
