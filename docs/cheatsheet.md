# Шпаргалка к защите (Optuna v5.0.0, сэмплеры)

## 1. Главная мысль за 30 секунд
`Study` хранит **стратегию** `sampler: BaseSampler`. Пользовательская функция вызывает `trial.suggest_*`.
`Trial._suggest` решает, откуда взять значение. Вариантов четыре:
1. **fixed**: параметр задан через `enqueue_trial` или `ask(fixed_distributions)`;
2. **single**: в распределении всего одно возможное значение;
3. **relative**: совместное сэмплирование всех параметров сразу, вызывается один раз на trial;
4. **independent**: сэмплирование одного параметра.

Значение записывается в `storage.set_trial_param`.
После завершения trial вызывается `sampler.after_trial`, затем `storage.set_trial_state_values`.
Сэмплер сам ничего не хранит: всю историю он читает через `study._get_trials()` из storage.

## 2. Путь выполнения (номера строк для тега v5.0.0)
| Шаг | Где | Что |
|---|---|---|
| 1 | `study/study.py` `Study.__init__` (~92) | `self.sampler = sampler or samplers.TPESampler()` |
| 2 | `study/_optimize.py` `_run_trial` | `trial = study.ask()` → `func(trial)` → `_tell_with_warning` |
| 3 | `study/study.py` `Study.ask` (514) | `_pop_waiting_trial_id()` (enqueue) или `storage.create_new_trial` → `Trial(self, trial_id)` |
| 4 | `trial/_trial.py:66` | `sampler.before_trial(...)` |
| 5 | `trial/_trial.py:602` `_suggest` | fixed → single → relative → independent |
| 6 | `trial/_trial.py:74-83` `relative_params` | ленивое свойство: `infer_relative_search_space` + `sample_relative` |
| 7 | `trial/_trial.py:621` | `sampler.sample_independent(...)` |
| 8 | `study/_tell.py:171` | `sampler.after_trial(...)`, затем `storage.set_trial_state_values` |

## 3. Как работает TPE (если спросят)
- Пока завершённых trial'ов меньше `n_startup_trials` (по умолчанию 10), `_sample_relative` возвращает `{}`. Тогда `sample_independent` делегирует вызов во вложенный `RandomSampler`.
- Дальше trial'ы делятся на below (лучшие γ(n) = min(ceil(0.1·n), 25)) и above (остальные).
- По каждой группе строится `_ParzenEstimator` (смесь усечённых нормальных распределений): l(x) по below и g(x) по above.
- Из l(x) берутся 24 кандидата (`n_ei_candidates`). Выбирается тот, у которого максимально log l(x) − log g(x).
- В v5.0.0 `multivariate` по умолчанию True для однокритериальной задачи. Поэтому relative search space — это пересечение пространств всех trial'ов (`IntersectionSearchSpace`).
- `constant_liar=True` по умолчанию. Running-trial'ы тоже учитываются в выборке above, чтобы параллельные воркеры не брали одну и ту же точку. Relative-параметры пишутся в `system_attrs["tpe:relative_params:i"]`.

## 4. Типовые просьбы ассистента и как их выполнить
| Просьба | Что поменять | Что изменится в трассе / какая часть библиотеки отвечает |
|---|---|---|
| Сменить алгоритм | `--sampler random` или `create_study(sampler=RandomSampler())` | `infer_relative_search_space` → ∅, всё идёт через `sample_independent`. Study и Trial не меняются: это Strategy |
| Убрать многомерный TPE | `--sampler tpe-independent` (`multivariate=False`) | relative space пуст. После startup каждый параметр сэмплируется отдельно через `TPESampler._sample({name: dist})` |
| Раньше включить TPE | `--n-startup-trials 3` | `sample_relative` начнёт возвращать значения с trial #3 |
| Задать стартовую точку | `--enqueue x=2,y=-1,kind=b` / `study.enqueue_trial(...)` | `Study.enqueue_trial` создаёт WAITING-trial с `system_attrs["fixed_params"]`. `ask()` берёт его через `_pop_waiting_trial_id`. В `_suggest` срабатывает ветка `_is_fixed_param`: сэмплер вызывается, но для этих параметров не используется |
| Зафиксировать параметр навсегда | `--fix kind=b` (`PartialFixedSampler`) | Декоратор убирает `kind` из relative space, а в `sample_independent` возвращает фиксированное значение |
| Максимизировать вместо минимизации | `--direction maximize` | `_split_trials` сортирует trial'ы с учётом `study.direction` |
| Условное (динамическое) пространство | `--objective dynamic` | В пересечении остаётся только `classifier`. `svc_*` и `rf_*` идут в `sample_independent`. Решение: `--sampler tpe-group` (`_GroupDecomposedSearchSpace`) |
| Параллельно | `--n-jobs 2` | `ThreadPoolExecutor`; `reseed_rng()` в каждом потоке (`_optimize_sequential(reseed_sampler_rng=True)`) |
| Свой сэмплер | см. `local_search_sampler.py` | Наследуем `BaseSampler`, реализуем 3 абстрактных метода. Хуки `before_trial`/`after_trial`/`reseed_rng` опциональны |
| Добавить параметр | новая строка `trial.suggest_int("z", 0, 5)` в objective | При multivariate `z` попадёт в relative space со следующего trial: пересечение пересчитывается каждый trial |
| Поменять радиус / поведение LocalSearch | `--radius 0.05`, правка `_perturb` или `after_trial` | |

## 5. Вопросы «почему так?»
- **Зачем два метода сэмплирования?** Define-by-run: пространство поиска заранее неизвестно, оно появляется по мере вызовов suggest. Relative работает для того, что уже стабильно встречалось в прошлых trial'ах (пересечение): там можно учитывать корреляции (GP, CMA-ES, multivariate TPE). Independent — запасной путь для новых или условных параметров.
- **Почему relative вызывается лениво, а не в `ask()`?** Комментарий в коде: «Evaluate it lazily to get as latest search as possible». Так учитываются trial'ы, завершившиеся параллельно, пока objective готовился к работе.
- **Почему сэмплеры передают `study`, а не список trial'ов?** Сэмплеру нужны направления оптимизации, storage (для system attrs) и кэш trial'ов. Минус: сильная связность, сэмплер зависит от приватного `study._get_trials` и `study._storage`.
- **Почему состояние хранится в storage?** Распределённая оптимизация: много процессов, один RDB или Journal. Состояние в памяти объекта сэмплера там потерялось бы. CMA-ES хранит состояние оптимизатора в `system_attrs`, TPE хранит relative_params. Мой `LocalSearchSampler` держит `_radius` в памяти, и при нескольких процессах радиус у каждого будет свой. Это сознательное ограничение.
- **Альтернатива:** ask-интерфейс сэмплера, который получает всё пространство заранее (define-and-run, как в Hyperopt/скриптах sklearn GridSearch). Проще, но без условных параметров и без ленивости.
