# localmodel backend: two engines

> ## ⚠️ LOCAL llama.cpp PATCH — RE-APPLY AFTER EVERY UPDATE
>
> `~/llama.cpp` carries a local 16-line patch that adds
> `--spec-draft-kv-offload` / `-kvod` (keep the draft KV cache on the GPU
> while `-nkvo` moves the main KV to RAM). **A `git pull` / engine update
> drops it.** After every update run `~/llama.cpp-patches/reapply.sh` (or
> follow `~/llama.cpp-patches/README.md`), otherwise the `qwen-chat` profile
> fails with "unknown argument: -kvod".
>
> Applied and built on `e30ad80fd` (0.9.0-dev, build 12155). Check with
> `cd ~/llama.cpp && git diff --stat` (3 files, +16 lines).

`localmodel` is a thin control plane in front of two inference engines. It does
not load weights itself; it decides *which* engine runs a model, starts it, and
watches it over HTTP.

```
                 ┌──────────────────────────────────────────┐
  localmodel ──► │ route(model)                             │
                 │   Strata family?  ──► strata launch      │
                 │   everything else ──► llama-server       │
                 └──────────────────────────────────────────┘
                        │                       │
                 OpenAI /v1 API          OpenAI /v1 API
                        │                       │
              (background server,        (child process,
               stopped with               owned PID, stderr
               `strata launch --stop`)    is the live log)
```

## Why two engines

> **2026-11-05:** the Strata engine was uninstalled from this machine (binary,
> packs, config and state removed); only the llama.cpp path is in use now. The
> routing described below still exists in the code.

Strata is not a general GGUF runner. Its public families are exactly three:

| family | packs |
| --- | --- |
| `qwen36` | Qwen3.6-35B-A3B: `ud-q4_k_xl`, `ud-q6_k_xl`, `q8_0` |
| `qwen` / `coder` / `swift` / `gyro` / `nvfp4` | Qwen3.8-Flash-Next and its variants |
| `deepseek41` | DeepSeek V4.1-Flash, 8K context only |

Everything else — Mixtral, Nemotron, Qwen3-30B-A3B, any dense model — runs on
llama.cpp. So "Strata for MoE, llama.cpp for dense" is the wrong split. The
correct split is **"Strata for its three families, llama.cpp for everything
else"**, and that is what `engine::supports_strata` implements.

Why Strata is still worth having: for its own families it does something
llama.cpp cannot. It keeps the routed experts it can fit in VRAM and computes
the rest from system RAM, and for Qwen3.6-35B-A3B it batches several
conversations in one decode window. On a 16 GB card that is the difference
between ~40 and ~60 tokens/s (published RTX 3060 12 GB numbers).

## Routing

```rust
engine::route(&model, forced)
```

1. `--engine strata|llama` wins, with a warning if the model is not a Strata
   family.
2. A Strata family name plus an installed `strata` binary selects Strata.
3. Otherwise llama.cpp.

`localmodel list` prints the resolved engine per file.

## Strata lifecycle

`strata launch --no-tui --yes --output json` starts the server, prints the
address and returns. The server keeps running in the background, so localmodel
never owns a PID there:

| action | localmodel |
| --- | --- |
| start | `strata launch … --yes --no-tui --output json` |
| address | parsed from the JSON (`api_url` / `url` / `base_url`) |
| live stats | poll `/metrics`, deltas of `tokens_predicted_total` |
| stop | `strata launch --stop` |
| installed packs | `strata models list --installed` |

`run_strata()` writes the launch output to `~/models/.localmodel-strata-<port>.log`
so the dashboard's log panel still has something to show.

For a MoE GGUF already on disk, `localmodel run <file>` passes it as
`--model-path`; only the three families above are accepted by Strata.

## llama.cpp lifecycle

Unchanged: `build_args()` builds the command, the child's stderr is streamed
into `~/models/.localmodel-<port>.log` and parsed for `eval time` lines, and
localmodel owns and kills the PID.

## Sizing model (`src/fit.rs`)

The numbers come from Strata's docs, not from guessing:

* Qwen3.6-35B-A3B — 45 KB/token of KV, **fp32 only**: this engine path has no
  `--kv` switch (`setup.py` warns and ignores it), so 128K costs ~5.9 GB of VRAM,
  not 1.5 GB. The hybrid mode also allocates a host arena for *every* routed
  expert (`cudaHostAlloc`, falling back to `malloc`), so the file's expert bytes
  must fit in system RAM: ~27 GB for UD-Q6_K_XL, ~18 GB for UD-Q4_K_XL. Strata's
  own planning figures are 44 GB and 32 GB of RAM respectively. Whatever experts
  the card holds after KV, the 0.77 GiB reserve and ~1 GB of buffers are pinned
  in VRAM; the rest are computed on the CPU from that host arena.
* Qwen3.8-Flash-Next — shard 1's experts must sit in RAM, plus ~10 GB for the
  system. Low-RAM mode (experts mmap'ed from the SSD) needs 32 GB RAM and a
  24 GB card, or the Coder pack.
* Gyro — one figure for VRAM + RAM together.

`localmodel doctor` prints the estimate per pack for this machine, split into a
memory verdict and a disk verdict, because they fail independently.

### Verified on the dev machine

RTX 4060 Ti 16 GB + Ryzen 5 5600 + 30.3 GB RAM, Strata 0.1.55:

* `qwen36/ud-q4_k_xl` at 128K: ready, 18.75 GiB pinned host arena, 11 of 256
  experts per layer in VRAM, 41 tok/s decode and ~100 tok/s prompt read on a
  short chat. Needs `STRATA_Q36_CHUNK=512` (or `--prefill 512` in the run
  config): the default chunk 2048 leaves only 1.61 GiB free at 128K and the
  engine refuses with "not enough VRAM ... 1760 MiB needed".
* `qwen36/ud-q6_k_xl` at 128K: the host arena alone is 27 GB; with the OS the
  box OOMs. It is marked `no` for a reason, not conservatively.
* Strata's `/metrics` is a JSON snapshot (`engine`, `live`, `requests`), while
  llama.cpp speaks Prometheus text. `stats.rs` parses both.
* Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp on llama.cpp: the qwen35 architecture runs
  every 4th layer as full attention (`full_attention_interval=4`, 65 blocks ->
  16 KV layers, 4 KV heads x 256 dim), so KV is 34 KiB/token at q8_0 and
  18 KiB at q4_0. Weights are 11.29 GiB, so q4_0 KV is the only option that
  keeps the GPU fully loaded.
* Measured Qwen3.8 configurations on the 4060 Ti (16 GB, desktop ~0.7 GiB):
  - **100K (102400) + MTP, q4_0 KV in VRAM (chosen):** with a browser open,
    15.43 GiB used / 523 MiB free at load, 321 MiB free after a 75K-token
    prompt. 36 tok/s short, ~29 tok/s at 18K depth, ~12 tok/s at 75K depth;
    prefill ~800 tok/s (16K) / ~484 tok/s (75K). Acceptance ~53%.
    112K + MTP loads too but leaves only ~190-430 MiB free (OOM risk),
    and 128K + MTP does not fit (OOM while creating the draft context).
  - 128K, q4_0 KV, no MTP: 20.5 tok/s, ~1.5 GiB free. This was the previous
    choice, rejected in favour of the MTP speed-up.
  - 192K, q4_0 KV in VRAM: loads with 71 MiB free but OOMs on the first
    4K-token request (CUDA error: out of memory). Not usable.
  - 262K, `-nkvo` (f16 main KV in RAM, ~17 GiB) + MTP with q4_0 draft KV
    also in RAM: 14.35 GiB VRAM, 13.1 tok/s, prefill ~413 tok/s. Without MTP:
    11.0 tok/s.
    The same model at 128K with q4_0 KV in VRAM prefills at ~906 tok/s and
    decodes at 20.2 tok/s at 4K depth, so `-nkvo` roughly halves throughput.
    Kept documented, not enabled.
* MTP is built into the ISTA-DASLab `-mtp` builds (15 nextn tensors,
  `qwen35.nextn_predict_layers = 1`). localmodel detects the `nextn` tensors
  and launches `--spec-type draft-mtp --spec-draft-n-max 3` with no
  `--spec-draft-model`: llama.cpp builds the draft context from the target
  model itself. The old standalone `mtp-Qwen3.8-27B-Q4_0.gguf` was redundant
  and lower quality, and cost ~775 MiB extra VRAM.
  Measured on the 4060 Ti, 32K ctx, q4_0 KV: 40 tok/s with `-b 2048`, 44 tok/s
  with `-b 512`; acceptance 0.68-0.79, mean len 3.0-3.35. n-max 1 gives ~30
  tok/s (acceptance 0.87, mean len 1.87), so 3 is the default for built-in MTP.
* Built-in MTP memory at 32K: 13.87 GiB process, ~1.35 GiB free. It costs
  +1.04 GiB over no MTP: +335 MiB for the `blk.64` weights, +449 MiB recurrent
  state (n-max 3 vs 150 MiB idle), +128 MiB draft KV (f16) and +130 MiB draft
  compute. At high context the draft KV scales with the main one, so the
  settings pass `-ctkd q4_0 -ctvd q4_0` (`--spec-draft-type-k/v`); that is
  what makes 112K + MTP fit. At 32K the difference was a wash.
* Chosen config on this box: 100K (`ctx = 102400`) with `-b 1024 -ub 256`,
  MTP on, q4_0 KV everywhere. Tested with a browser open (the usual dsh
  scenario): ~0.52 GiB free at load and ~0.32 GiB after 75K tokens of prompt,
  no OOM. There is no iGPU (Ryzen 5 5600, one 4060 Ti), so the desktop cannot
  be moved off the card.
* **Two Qwen profiles.** The default entry is the AGENT config above (100K,
  fast, all KV in VRAM). `[model."qwen-chat"]` (`applies_to` the same GGUF,
  launched with `localmodel run Qwen3.8 --profile qwen-chat`) is the
  long-context chat profile: weights + draft KV on the GPU, the whole main KV
  in RAM (`-nkvo`), both caches q8_0, draft KV kept on the GPU by the local
  `-kvod` patch. **262K is the chat default** and was verified on
  0.9.0-dev build 12155: `n_ctx = 262144`, 15.2 GiB VRAM used / 722 MiB free,
  13 GiB RAM, 12.9 tok/s, GPU 63-69% / 83 W while generating, draft
  acceptance ~50% (147/294). The MTP draft KV grows with context
  (~13 KiB/token, ~3.4 GiB at 262K); the main q8_0 KV in RAM is ~8.5 GiB.
  The 722 MiB VRAM headroom covers the desktop; if a game/video takes the
  GPU, fall back to `ctx = 131072` (still q8_0/q8_0: 13.5 GiB VRAM,
  2.4 GiB free, 12.2 tok/s). Earlier 128K f16/q4 numbers (13.37 GiB VRAM,
  13.1 tok/s) are obsolete.
* Ornith-1.5-35B-A3B-AD-Q4_K-IQ4_XS (18.74 GiB, 40 layers) runs at the full
  262K with q8_0 KV in VRAM using `ncmoe = 25` (experts of the first 25
  layers in RAM) and `-ngl 99`. The AD file has no `nextn` tensors; the MTP
  head is a separate 1.85 GiB file, wired in with `--spec-type draft-mtp
  --spec-draft-model mtp-Ornith-1.5-35B-A3B-Q8_0.gguf --spec-draft-n-max 6
  --spec-draft-ngl 999`. Measured with a browser open: 15.0 GiB used,
  ~0.95 GiB free, 29.7 tok/s warm (23.2 with a cold page cache for the
  CPU-resident experts), acceptance ~30%. `ncmoe = 28` also works but slower.
* Ornith-1.5-9B-AD-Q8_0-Q6_K (7.96 GiB) has no MTP head in the file or repo,
  so it runs plain at 262K with q8_0 KV: 13.4 GiB used, ~2.5 GiB free with a
  browser, 33 tok/s.
* `--cache-reuse 256` is on for every llama.cpp launch. With a long shared
  system prompt the second request costs 6.6 s instead of 11.6 s (prefill of
  the 2.4K prefix is skipped); llama.cpp's `prompt_per_second` metric looks
  low in that case because it divides the whole prompt by the short reuse
  time, so judge by wall clock.
* `--tools all` (llama.cpp built-in agent tools) is available as `tools` in
  settings / `--tools`, off by default: the server has no API key.
* Reasoning effort: localmodel parses the GGUF chat template and passes only
  the variables the template understands (`src/jinja.rs`): Qwen3.8 gets
  `reasoning_effort` (xhigh/medium/low), Nemotron gets `medium_effort` /
  `enable_thinking`. House defaults: Qwen-style = medium (xhigh overthinks),
  Nemotron-style = full thinking. Override per model with `effort = "..."` in
  settings.toml, per launch with `--effort`, or per message in the REPL with
  `/effort`. LiLiCorr/MTP draft depth: `draft_tokens` in settings
  (Nemotron default 7, NVIDIA's recipe).

## Adding a family

1. Add the pack to `fit::CATALOG` with its published download size, expert size
   and planned RAM.
2. Teach `engine::supports_strata` the model name.
3. `localmodel strata <family> <size>` then works with no further code: the
   family and size are passed straight to `strata launch --family --model`.
