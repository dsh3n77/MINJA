#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Full experiment sweep for the persistence-under-eviction memory-poisoning study.
#
# Three blocks:
#   A. PERSISTENCE sweep  (retrieval only, 0 API calls): tag none/generic/ours/keyword
#        -> SSR(t)/BIR(t) survival curves + survival-vs-collateral trade-off.
#   B. GATE ablation (needs LLM): 8 templates (V1/V2 x plain/note/imperative/system)
#        at the keyword operating point (max exposure = hardest test for the gate)
#        -> ASR (fires on victim) and BLR (leaks on benign).
#   C. GATE value (needs LLM): keyword tag, UNGATED redirect -> BLR blows up,
#        showing what the gate buys.
#
# Usage:
#   MOCK (no API key, offline, instant -- proves the pipeline + output format):
#       ./run_all.sh --mock
#   REAL (your environment; needs OPENAI_API_KEY and the all-MiniLM-L6-v2 embedder):
#       export OPENAI_API_KEY=sk-...
#       ./run_all.sh --model gpt-4o
#
# Extra args after the first are passed through to every run, e.g.:
#       ./run_all.sh --model gpt-4o --policy lfu --victims 5
# ---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"
export PYTHONHASHSEED=0          # reproducibility

COMMON="--dataset mimic_iii --L 80 --T 250 --K 4 --measure_every 50 --victims 3 $@"
# behavioural probe sizes (drive API cost: (n_victim_probe + n_benign_probe) x victims per run)
BEHAV="--n_victim_probe 6 --n_benign_probe 15"

mkdir -p results
rm -f results/tag__*.json results/gate__*.json

echo "########## A. PERSISTENCE sweep (retrieval only, no API) ##########"
for tag in none generic ours keyword; do
  python3 run_experiment.py $COMMON --no_behavior \
    --tag_kind $tag --gate gated --gate_variant V1 --gate_style plain \
    --tag "tag__$tag" --out "results/tag__$tag.json"
done

echo "########## B. GATE ablation: 8 templates @ keyword operating point ##########"
for variant in V1 V2; do
  for style in plain note imperative system; do
    python3 run_experiment.py $COMMON $BEHAV \
      --tag_kind keyword --gate gated --gate_variant $variant --gate_style $style \
      --tag "gate__${variant}_${style}" --out "results/gate__${variant}_${style}.json"
  done
done

echo "########## C. GATE value: keyword tag, UNGATED redirect ##########"
python3 run_experiment.py $COMMON $BEHAV \
  --tag_kind keyword --gate none \
  --tag "gate__ungated" --out "results/gate__ungated.json"

echo "########## summarise + plot ##########"
python3 plot_results.py

echo "DONE. See results/summary.csv and results/*.png"
