import argparse

import optuna

from common import objective, quiet_optuna
from tracing import TracingSampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", choices=["random", "tpe"], default="tpe")
    args = parser.parse_args()
    quiet_optuna()

    if args.sampler == "random":
        base_sampler = optuna.samplers.RandomSampler(seed=67)
    else:
        base_sampler = optuna.samplers.TPESampler(
            seed=67, n_startup_trials=3, multivariate=True
        )

    study = optuna.create_study(
        direction="minimize", sampler=TracingSampler(base_sampler)
    )
    study.optimize(objective, n_trials=15)

    print(f"Лучшее: x={study.best_params['x']:.3f} y={study.best_params['y']:.3f} f={study.best_value:.3f}")


if __name__ == "__main__":
    main()
