# localmodel

Терминальный пульт для локальных GGUF-моделей: находит файлы в `~/models`,
собирает команду `llama-server`, считает память, запускает сервер и показывает
живой дашборд. Настройки хранятся в одном `settings.toml` и редактируются как
с клавиатуры, так и из скриптов.

```
localmodel                 # выбрать модель из списка и запустить
localmodel hub             # модели + их настройки на одном экране
localmodel dev             # песочница: крутить ручки, запускать, смотреть лог
localmodel run <model>     # запустить по имени
localmodel doctor          # что влезает в эту машину
```

## Возможности

- **Один экран для модели и её настроек.** `hub` показывает только те ручки,
  которые вообще применимы к выбранной модели (MoE, MTP/draft, vision), их
  эффективные значения, источник каждого значения и живой прогноз VRAM.
- **Провенанс вместо «магии».** У каждого параметра есть источник:
  `cli > model > defaults > builtin > derived`. `--dry-run --explain` и
  `settings --show --origin` печатают ту же таблицу без запуска сервера.
- **Безопасное сохранение.** `settings.toml` пишется через `toml_edit`:
  комментарии, порядок ключей и профили, которые не трогали, остаются как были.
- **Калькулятор VRAM по GGUF.** Парсятся метаданные и таблица тензоров: слои,
  головы KV, `head_dim`, sliding window, nextn/MTP, эксперты. Отсюда —
  `ctx max`, вердикт `fits / tight / no` и предложение контекста.
- **Раскладка по памяти.** `src/mem.rs` повторяет правила llama.cpp:
  `-ngl` (последние N блоков), `--n-cpu-moe` (эксперты первых N),
  `token_embd` в RAM, `--no-kv-offload`, `--split-mode` и `--tensor-split`
  для двух карт. После загрузки прогноз заменяется фактической раскладкой из
  лога llama.cpp (`--log-verbosity 4`).
- **Живой дашборд.** Токены/с, prefill, draft acceptance по позициям, GPU
  (util/температура/питание), RAM, лог сервера. `p` замораживает экран, чтобы
  можно было выделить и скопировать текст.
- **Песочница `dev`.** Любые ручки меняются в памяти, сервер запускается тут же;
  `q` останавливает и откатывает всё, `s` сохраняет удачный эксперимент в
  профиль модели.
- **MTP и внешние draft-модели.** Встроенная MTP-голова (`blk.N.nextn.*`)
  определяется автоматически; рядом лежащий `mtp-*.gguf` / `draft-*.gguf`
  подхватывается как внешний драфт. Для vision ищется `mmproj-*.gguf`.
- **Reasoning effort из шаблона.** `src/jinja.rs` разбирает chat template и
  передаёт только те переменные, которые он понимает (`reasoning_effort`,
  `medium_effort`, `enable_thinking`), а не одной кучей.
- **Два движка.** По умолчанию llama.cpp; Strata подключается для своих
  семейств (Qwen3.6-35B-A3B, Qwen3.8-Flash-Next, DeepSeek V4.1-Flash), если
  бинарь установлен. `--engine auto|llama|strata` переопределяет выбор.

## Экраны

### `localmodel hub`

```
┌ models ──────────────────┐┌ Ornith-1.5-9B-Q6_K ─────────────────────────────┐
│ ● Ornith-1.5-9B-Q6_K     ││ 7.0G Q6_K · 33L · MTP · qwen35 · dense           │
│   Qwen3.8-27B-GSQ-...    ││ VRAM 13.7 / 15.5 GiB  ████████████░░  fits        │
│   gemma-4-12b-it         ││ ctx max для этой карты: 262144                   │
│                          ││                                                  │
│ + добавить с HuggingFace ││ CONTEXT                                          │
│                          ││  context size        262144    model             │
│                          ││  KV cache K          q8_0      default           │
│                          ││ SPEED                                            │
│                          ││  draft tokens        2         model             │
│                          ││  cache reuse         off       builtin (SSM)     │
└──────────────────────────┘└──────────────────────────────────────────────────┘
 ↑↓ выбор · Tab панель · Enter править · − сбросить · d argv · s сохранить · L запуск
```

- `Enter` — открыть настройки модели; на строке `+ добавить` — вставить URL
  с HuggingFace.
- `Enter` на ручке — bool переключается, enum листается, остальное открывает
  строку ввода с проверкой типа и границ.
- `−` снимает override, `s` пишет только изменённые ключи, `d` показывает
  argv с происхождением, `L` сохраняет и запускает, `x` удаляет `.gguf`.
- Без TTY (`localmodel hub | cat`) печатает список моделей и выходит — скрипты
  пользуются `settings` и `run --dry-run`.

### `localmodel dev`

Песочница поверх той же модели данных. Правки живут только в памяти и никогда
не попадают в `settings.toml` сами по себе.

```
↑↓ ручка · Enter править · s сохранить профиль · m другая модель
r запуск/перезапуск · Tab лог · q остановить и откатить · ? помощь
```

После удачного запуска `s` сохраняет только тронутые ключи в
`[model."ИМЯ"]`, остальное продолжает приходить из `[defaults]`.

### Дашборд запущенной модели

```
?  h    помощь
l       показать / скрыть лог
p       пауза: экран замирает, текст можно выделять и копировать
o       продолжить
r       перезапустить сервер
q Esc   остановить и выйти
s       подсказка, где живут настройки (это hub)
```

Карточка сама выбирает раскладку под высоту терминала: полная, компактная или
минимальная. Ширина ограничивается так, чтобы строки не переносились и экран
не «размазывался».

## Установка

Нужны:

- Linux (используются `/proc` и, для статистики GPU, `nvidia-smi`);
- Rust и Cargo;
- собранный `llama-server` из [llama.cpp](https://github.com/ggml-org/llama.cpp);
- `curl` для `localmodel add`.

По умолчанию бинарь ищется в `~/llama.cpp/build/bin/llama-server`, модели — в
`~/models`. Оба пути переопределяются переменными окружения (см. ниже).

```bash
git clone git@github.com:Denisssy/localmodel.git
cd localmodel

make install          # cargo build --release + install в ~/.local/bin/localmodel
# или
cargo build --release
install -m 755 target/release/localmodel ~/.local/bin/localmodel
```

`make install` — единственный поддерживаемый способ обновить бинарь на PATH:
запуск устаревшей копии из `~/.local/bin` молча теряет новые функции.

## Быстрый старт

```bash
# 1. положить .gguf в ~/models (или указать LOCALMODEL_DIR)
localmodel list

# 2. посмотреть, что вообще влезает в машину
localmodel doctor

# 3. открыть хаб, выбрать модель, при желании поправить ctx/KV и нажать L
localmodel hub

# 4. или сразу запустить по подстроке имени
localmodel run Ornith

# 5. перед запуском посмотреть точную команду и происхождение флагов
localmodel run Ornith --dry-run --explain
```

Скачивание по прямой ссылке:

```bash
localmodel add https://huggingface.co/<repo>/resolve/main/<file>.gguf
localmodel add <url> --name my-model.gguf
```

## Команды

| Команда | Что делает |
| --- | --- |
| `localmodel` | выбор модели из списка (на TTY) или `list` |
| `localmodel hub` | интерактивный экран: модели, настройки, argv, запуск |
| `localmodel dev` | песочница с откатом и сохранением профиля |
| `localmodel run <model>` | запустить модель по подстроке имени или индексу |
| `localmodel run <family>/<size>` | запуск Strata-пака |
| `localmodel strata <family> <size>` | запуск через Strata |
| `localmodel list` | список скачанных моделей и запущенных серверов |
| `localmodel info <model>` | карточка модели без запуска |
| `localmodel doctor` | железо и вердикты по каталогу моделей |
| `localmodel params [pattern]` | шпаргалка по флагам llama.cpp / grep по `--help` |
| `localmodel settings` | показать и изменить `settings.toml` |
| `localmodel add <url>` | скачать `.gguf` в папку моделей |
| `localmodel stop` | остановить запущенные серверы |
| `localmodel version` | версия |

### Ключевые флаги `run`

| Флаг | Смысл |
| --- | --- |
| `--ctx N` | размер контекста (по умолчанию 131072) |
| `--port N` | порт (по умолчанию первый свободный с 8080) |
| `--kv T` | тип KV-кэша сразу для K и V |
| `--ctk T`, `--ctv T` | тип KV-кэша по отдельности |
| `--draft-ctk T`, `--draft-ctv T` | KV для draft/MTP-контекста |
| `--ngl N` | сколько слоёв на GPU |
| `--ncmoe N` | сколько первых MoE-слоёв оставить экспертами в RAM |
| `--kv-ram` / `--kv-vram` | держать основной KV в RAM или на карте |
| `--split-mode M` | `layer`, `row`, `tensor`, `none` для двух карт |
| `--tensor-split S` | доли карт, например `0.57,0.43` |
| `--draft-tokens N` | сколько токенов предлагает draft/MTP |
| `--draft-gpu` | внешний draft на GPU (встроенная MTP это игнорирует) |
| `--reasoning on\|off` | включить/выключить reasoning |
| `--effort L` | `high`, `xhigh`, `medium`, `low`, `off` |
| `--tools L` | встроенные агентские инструменты llama.cpp |
| `--vision on\|off`, `--mmproj FILE` | мультимодальный вход |
| `--profile NAME` | взять именованный профиль из настроек |
| `--extra "..."` | произвольные флаги llama.cpp (добавляются последними) |
| `--dry-run` | напечатать команду и выйти |
| `--explain` | таблица происхождения значений + команда |
| `--strict` | неизвестный или затенённый флаг в `extra` — ошибка |
| `--repl` | чат в терминале вместо дашборда |

Полный список: `localmodel help`, флаги самого движка: `localmodel params --all`.

## Настройки

Файл: `~/.config/localmodel/settings.toml` (или `$XDG_CONFIG_HOME/localmodel/settings.toml`).

```toml
[defaults]
ctx = 131072
ctk = "q8_0"
ctv = "q8_0"

[model."Ornith-1.5-9B-Q6_K"]
ctx = 262144
draft_tokens = 2

# именованный вариант той же модели: localmodel run Qwen --profile qwen-chat
[model."qwen-chat"]
applies_to = "Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp"
ctx = 262144
ctk = "q8_0"
ctv = "q8_0"
```

Приоритет значений: **CLI → `[model.NAME]` → `[defaults]` → встроенный
дефолт → производное значение** (например, `ctv` наследует `ctk`, а
draft-параметры зависят от класса модели).

Без TUI:

```bash
localmodel settings                                   # показать файл как есть
localmodel settings --defaults --set ctx=131072
localmodel settings --model Ornith --set ctx=262144 --unset ctk
localmodel settings --model Ornith --show --origin    # откуда каждое значение
```

`--set` проверяет имя ключа, тип и границы; неизвестный ключ — ошибка, а не
молчаливое игнорирование. `--unset` убирает override, возвращая значение
следующего слоя. Сохранение меняет только тронутые ключи.

## Память и раскладка

Три источника, которые дополняют друг друга:

1. **GGUF.** `src/gguf.rs` читает метаданные и таблицу тензоров: `block_count`,
   `head_count_kv`, `key_length`/`value_length`, `full_attention_interval`,
   sliding window, `nextn_predict_layers`, размеры экспертов. Размер считается
   по типу тензора, а не по размеру файла.
2. **Планировщик.** `src/mem.rs` раскладывает веса, KV, compute и CUDA-контекст
   по картам и RAM по тем же правилам, что llama.cpp.
3. **Лог загрузки.** При `--log-verbosity 4` llama.cpp печатает
   `memory breakdown` и строки `model/KV/compute buffer size`. `src/telemetry.rs`
   разбирает их и показывает как `actual · llama.cpp startup log`, заменяя
   прогноз фактом. Это отчёт о стартовой аллокации, а не live-телеметрия.

Отсюда `localmodel info`, стартовая карточка, дашборд и шапка хаба показывают
одно и то же: сколько весов, сколько KV, где он лежит, какие слои на какой
карте и сколько осталось запаса.

## Модели и движки

- **llama.cpp** — основной путь для любых GGUF: dense, MoE, гибридные SSM,
  vision, MTP.
- **Strata** — только для своих семейств. Если бинарь не установлен,
  маршрутизация остаётся на llama.cpp; `--engine` позволяет выбрать вручную.
- **MTP.** Встроенная голова определяется по тензорам `blk.N.nextn.*` —
  отдельный draft-файл не нужен. Внешний драфт ищется рядом с моделью
  (`mtp-*.gguf`, `draft-*.gguf`, `lilicorr-*.gguf`).
- **Vision.** `mmproj-*.gguf` рядом с моделью включает режим vision и
  показывает соответствующие ручки в хабе.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `LOCALMODEL_DIR` | `~/models` | папка с `.gguf` |
| `LLAMA_SERVER` | `~/llama.cpp/build/bin/llama-server` | бинарь llama.cpp |
| `XDG_CONFIG_HOME` | `~/.config` | где лежит `localmodel/settings.toml` |

Сервер по умолчанию слушает `127.0.0.1` и не требует API-ключа. Если нужен
доступ из сети, задайте `host`/`api_key` в профиле или флагами `--host` и
`--api-key`.

## Разработка

```bash
cargo build --release
cargo test          # 70 тестов: argv golden, KV-математика, раскладка, TOML
make install        # обновить ~/.local/bin/localmodel
```

Устройство:

```
src/knobs.rs      реестр настроек — единственный источник правды
src/resolve.rs    cli/model/defaults/builtin → значение + источник
src/settings.rs   чтение и точечная запись settings.toml
src/gguf.rs       метаданные и таблица тензоров GGUF
src/fit.rs        «поместится ли»: VRAM, ctx max, каталог
src/mem.rs        «что где лежит»: карты, RAM, слои, эксперты
src/telemetry.rs  фактическая раскладка из лога llama.cpp
src/hub.rs        экран моделей и настроек
src/dev.rs        песочница
src/main.rs       CLI, сборка argv, дашборд, карточки
```

`tests` внутри `main.rs` фиксируют argv для реальных конфигураций побайтово:
рефакторинг не должен менять команду, пока это не сделано намеренно.

## Документация

- [`docs/SETTINGS-PLAN.md`](docs/SETTINGS-PLAN.md) — как устроены реестр,
  провенанс, калькулятор и хаб; история фаз.
- [`docs/BACKEND.md`](docs/BACKEND.md) — два движка, жизненный цикл серверов,
  маршрутизация и замеры на конкретной машине.
