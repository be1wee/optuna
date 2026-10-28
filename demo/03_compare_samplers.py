"""Сравнение стратегий на одной и той же целевой функции.

Меняется ТОЛЬКО объект сэмплера — Study, Trial и целевая функция не трогаются.

  python demo/03_compare_samplers.py
  python demo/03_compare_samplers.py --objective rf --n-trials 30 --samplers random tpe local
"""

from __future__ import annotations

import importlib.util
import statistics
import time

import optuna
from common import OBJECTIVES, SAMPLER_NAMES, base_arg_parser, make_sampler, quiet_optuna


def main() -> None:
    p = base_arg_parser(__doc__)
    p.set_defaults(n_trials=40)
    p.add_argument("--samplers", nargs="+", choices=SAMPLER_NAMES,
                   default=["random", "tpe", "cmaes", "gp", "local"])
    p.add_argument("--seeds", type=int, default=3, help="число повторов с разными seed")
    p.add_argument("--plot", default="compare.png", help="куда сохранить график ('' — не сохранять)")
    args = p.parse_args()
    quiet_optuna()
    optuna.logging.set_verbosity(optuna.logging.ERROR)  # без предупреждений CMA-ES о fallback для kind

    curves: dict[str, list[float]] = {}
    print(f"objective={args.objective}, n_trials={args.n_trials}, seeds={args.seeds}\n")
    print(f"{'sampler':<16}{'best (mean)':>14}{'best (min)':>14}{'time, s':>10}")
    for name in args.samplers:
        if name == "gp" and importlib.util.find_spec("torch") is None:
            print(f"{name:<16}  пропущен: GPSampler требует torch (см. README)")
            continue
        bests, runs = [], []
        t0 = time.perf_counter()
        for s in range(args.seeds):
            sampler = make_sampler(name, args.seed + s, args.n_startup_trials)
            study = optuna.create_study(sampler=sampler)
            study.optimize(OBJECTIVES[args.objective], n_trials=args.n_trials)
            bests.append(study.best_value)
            vals = [t.value for t in study.trials]
            runs.append([min(vals[: i + 1]) for i in range(len(vals))])
        dt = time.perf_counter() - t0
        curves[name] = [statistics.mean(c[i] for c in runs) for i in range(args.n_trials)]
        print(f"{name:<16}{statistics.mean(bests):>14.5f}{min(bests):>14.5f}{dt:>10.1f}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4.5))
        for name, c in curves.items():
            plt.plot(range(1, len(c) + 1), c, label=name)
        plt.axvline(args.n_startup_trials, color="gray", ls="--", lw=0.8)
        plt.yscale("log")
        plt.xlabel("trial")
        plt.ylabel("best value so far (mean over seeds)")
        plt.title(f"objective = {args.objective}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120)
        print(f"\nГрафик сохранён: {args.plot}")


if __name__ == "__main__":
    main()
