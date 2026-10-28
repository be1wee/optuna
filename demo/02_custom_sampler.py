"""Точка расширения: собственный LocalSearchSampler вместо встроенного.

  python demo/02_custom_sampler.py                    # трассировка LocalSearchSampler
  python demo/02_custom_sampler.py --objective rf
"""

from __future__ import annotations

import optuna
from common import OBJECTIVES, base_arg_parser, quiet_optuna
from local_search_sampler import LocalSearchSampler
from tracing_sampler import TracingSampler


def main() -> None:
    p = base_arg_parser(__doc__)
    p.set_defaults(n_trials=20, n_startup_trials=5)
    p.add_argument("--radius", type=float, default=0.2)
    args = p.parse_args()
    quiet_optuna()

    local = LocalSearchSampler(
        n_startup_trials=args.n_startup_trials, radius=args.radius, seed=args.seed
    )
    study = optuna.create_study(sampler=TracingSampler(local, show_caller=False))
    study.optimize(OBJECTIVES[args.objective], n_trials=args.n_trials)

    print("\n================ ИТОГ ================")
    print(f"best params : {study.best_params}")
    print(f"best value  : {study.best_value:.6f}")
    print(f"final radius: {local._radius:.4f}")


if __name__ == "__main__":
    main()
