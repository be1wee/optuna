import optuna

from common import objective, quiet_optuna


def main() -> None:
    quiet_optuna()
    sampler = optuna.samplers.RandomSampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    for _ in range(5):
        trial = study.ask()
        value = objective(trial)
        study.tell(trial, value)
        print(f"{trial.number + 1:2} | x={trial.params['x']:.3f} y={trial.params['y']:.3f} | f={value:.3f}")

    print(f"Лучшее: x={study.best_params['x']:.3f} y={study.best_params['y']:.3f} f={study.best_value:.3f}")


if __name__ == "__main__":
    main()
