# Persistence under eviction — EHR agent (built on MINJA)

Attack experiment: a **single** poisoned memory record that (a) **persists** in a
bounded, usage-evicting long-term memory (LRU/LFU), and (b) carries a
**conditional gate** that redirects on the victim entity while staying inert on
everything else (so benign utility is preserved). Built directly on MINJA's EHR
agent (`EHR/ehragent`), reusing its retriever, its knowledge-generation prompt,
its MIMIC-III / eICU data, and its attack-success notion.

```
persist_attack/
  gate_templates.py   # 8 gated templates (V1/V2 x plain/note/imperative/system), MIMIC + eICU, PSS
  persist_lib.py      # retriever (MiniLM->TFIDF fallback), DynamicMemory (LRU/LFU), LLM backend, judge, data
  run_experiment.py   # one (tag,gate) configuration -> results JSON
  run_all.sh          # full sweep: persistence + 8-template gate ablation + gate-value
  plot_results.py     # summary.csv + 4 figures
  results/            # outputs
```

## What one run does

1. **Inject** one poisoned record into a memory of capacity `L`:
   `question = victim_question + TAG` (the retrieval key), `knowledge = payload`.
2. **Evict**: stream `T` benign EHR queries; each retrieves top-`K` (refreshing
   LRU/LFU usage) and appends a benign record, forcing eviction. `SSR(t)/BIR(t)`
   logged along the way.
3. **Measure** at the endpoint:

| metric | meaning | source | want |
|---|---|---|---|
| **SSR** | survival: poison still in top-K for **victim** queries | retrieval | high |
| **BIR** | benign intrusion: poison in top-K for **benign** queries | retrieval | low |
| **ASR** | generated *Knowledge* redirects V→T on **victim** queries | LLM | high |
| **BLR** | generated *Knowledge* wrongly redirects on **benign** queries (UD-proxy) | LLM | low |

`leak|exposed` = BLR restricted to benign queries that actually retrieved the
poison — the pure measure of gate leakage.

The 8 templates **share one retrieval key**, so their SSR/BIR are identical; they
differ only in **ASR** (fires on victim) and **BLR** (stays inert on benign).
That is the whole point of the gate ablation.

## Run it

### Offline, no API key (proves the pipeline + output format)
```bash
cd EHR/persist_attack
./run_all.sh --mock
```
`--mock` uses an offline simulator for the LLM and TF-IDF if the MiniLM embedder
can't be downloaded. **The mock's ASR/BLR numbers are simulated, not real** — use
it only to check that everything runs and to see the table/figure format.

### Real run (your environment)
```bash
pip install openai sentence-transformers scikit-learn matplotlib numpy
export OPENAI_API_KEY=sk-...
cd EHR/persist_attack
./run_all.sh --model gpt-4o          # all-MiniLM-L6-v2 downloads on first use
```
Any extra flags pass through to every run, e.g.
`./run_all.sh --model gpt-4o --policy lfu --victims 5`.

**API cost** is driven by `(n_victim_probe + n_benign_probe) × victims` per
behavioural run; `run_all.sh` uses `6 + 15` probes × `3` victims ≈ 63 calls/run,
× 9 behavioural runs ≈ **~570 calls**. The 4 persistence runs use `--no_behavior`
(retrieval only, **0** API calls). Scale probes/victims up for tighter estimates.

### One configuration by hand
```bash
python3 run_experiment.py --model gpt-4o \
    --tag_kind keyword --gate gated --gate_variant V1 --gate_style note \
    --policy lru --L 60 --T 400 --K 4 --victims 3
```

## The sweep (`run_all.sh`)

- **A. Persistence** (retrieval only): `tag_kind ∈ {none, generic, keyword, ours}`,
  gate fixed → `SSR(t)/BIR(t)` curves + survival-vs-collateral trade-off.
  `none` = vanilla MINJA (no tag) → dies under eviction.
- **B. Gate ablation** (LLM): the 8 templates at the **keyword** operating point
  (maximum benign exposure = hardest test for the gate) → ASR vs BLR.
- **C. Gate value** (LLM): keyword tag, **ungated** MINJA-style redirect → BLR
  blows up, quantifying what the gate buys.

Outputs: `results/summary.csv`, `results/fig_persistence.png`,
`fig_persistence_tradeoff.png`, `fig_gate_ablation.png`, `fig_gate_tradeoff.png`.

## Key knobs

| flag | default | note |
|---|---|---|
| `--dataset` | `mimic_iii` | or `eicu` (drug redirect) |
| `--tag_kind` | `ours` | `none`/`generic`/`keyword`/`ours` — sets the SSR/BIR operating point |
| `--gate` | `gated` | `none` = ungated redirect (vanilla / ours-without-gate) |
| `--gate_variant` / `--gate_style` | `V1` / `plain` | the 8 templates |
| `--inject_mode` | `craft` | `craft`: payload = crafted gate (single-record). `generate`: elicit R from the agent, MINJA-style |
| `--policy` | `lru` | or `lfu` |
| `--L` / `--T` / `--K` | `60` / `400` / `4` | capacity / background stream / top-K |
| `--retriever` | `auto` | `embed` (MiniLM) / `tfidf` / `auto` (embed→tfidf) |
| `--mock` | off | offline LLM simulator |
| `--no_behavior` | off | skip ASR/BLR (retrieval only, 0 API cost) |

## Full-agent fidelity (optional extension)

The default `--backend knowledge` measures ASR/BLR at the **knowledge-generation**
step (`prompts_mimic.RetrKnowledge`) — exactly where MINJA's poison takes effect,
one LLM call per probe, no DB needed. This is sufficient for the 8-template gate
ablation.

For execution-level utility drop (real answer accuracy on the MIMIC DB), the
higher-fidelity path runs MINJA's complete agent (`ehragent/main.py`, autogen +
`run_code` over the shipped CSVs) and scores with `attack_check.py`. Seam: build
the bounded `DynamicMemory` here, dump it to `long_term_memory` JSON via
`main.py --load_memory_path`, run the victim/benign probe set through the agent,
and diff the logs with `attack_check.py` (ASR) and `judge()` (accuracy → UD).
`--backend full_agent` is stubbed with this note.

## Notes

- **Mock numbers are illustrative only.** Real gate differences come from the real
  model. The mock exists to validate plumbing and output format with no API key.
- In the sandbox the MiniLM download is blocked, so mock runs fall back to TF-IDF;
  your environment downloads the real embedder and should use `--retriever embed`
  (the default `auto` does this automatically).
