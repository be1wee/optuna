"""Трассировка всех вызовов сэмплера во время study.optimize().

Примеры:
  python demo/01_trace_sampling.py                                # TPE, quadratic
  python demo/01_trace_sampling.py --sampler random
  python demo/01_trace_sampling.py --sampler tpe-independent      # multivariate=False
  python demo/01_trace_sampling.py --objective dynamic --n-trials 14
  python demo/01_trace_sampling.py --enqueue x=2,y=-1,kind=b      # фиксированные параметры
  python demo/01_trace_sampling.py --fix kind=b                   # PartialFixedSampler
  python demo/01_trace_sampling.py --n-jobs 2                     # потоки + reseed_rng
"""

from __future__ import annotations

import optuna

from common import OBJECTIVES, base_arg_parser, make_sampler, quiet_optuna
from tracing_sampler import TracingSampler


def _parse_kv(s: str | None) -> dict:
    result: dict = {}
    if not s:
        return result
    for item in s.split(","):
        k, v = item.split("=")
        try:
            result[k] = float(v) if "." in v or "e" in v else int(v)
        except ValueError:
            result[k] = v
    return result


def main() -> None:
    p = base_arg_parser(__doc__)
    p.add_argument("--direction", choices=["minimize", "maximize"], default="minimize")
    p.add_argument("--enqueue", help="параметры первого trial, напр. x=2,y=-1,kind=b")
    p.add_argument("--fix", help="зафиксировать параметры через PartialFixedSampler, напр. kind=b")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--no-caller", action="store_true", help="не печатать место вызова в optuna")
    args = p.parse_args()
    quiet_optuna()

    base = make_sampler(args.sampler, args.seed, args.n_startup_trials)
    if args.fix:
        base = optuna.samplers.PartialFixedSampler(_parse_kv(args.fix), base)
    sampler = TracingSampler(base, show_caller=not args.no_caller)

    study = optuna.create_study(direction=args.direction, sampler=sampler)
    if args.enqueue:
        study.enqueue_trial(_parse_kv(args.enqueue))

    study.optimize(OBJECTIVES[args.objective], n_trials=args.n_trials, n_jobs=args.n_jobs)

    print("\n================ ИТОГ ================")
    print(f"sampler      : {base}")
    print(f"best trial   : #{study.best_trial.number}")
    print(f"best params  : {study.best_params}")
    print(f"best value   : {study.best_value:.6f}")


if __name__ == "__main__":
    main()
