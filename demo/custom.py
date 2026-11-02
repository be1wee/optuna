import optuna

from common import objective, quiet_optuna
from local_sampler import LocalSearchSampler
from tracing import TracingSampler


def main() -> None:
    quiet_optuna()
    sampler = LocalSearchSampler(seed=42, n_startup_trials=3, radius=0.1)
    study = optuna.create_study(
        direction="minimize", sampler=TracingSampler(sampler)
    )
    study.optimize(objective, n_trials=15)

    print(f"Лучшее: x={study.best_params['x']:.3f} y={study.best_params['y']:.3f} f={study.best_value:.3f}")


if __name__ == "__main__":
    main()
