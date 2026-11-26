"""
Persistence-under-eviction memory-poisoning experiment on MINJA's EHR agent.

One run = one (tag, gate) configuration, on one or more victim entities:

  1. INJECT a single poisoned record into a bounded long-term memory.
       question  = victim_question + TAG        (the retrieval key)
       knowledge = payload R                     (the gate, or an ungated redirect)
     craft mode  : R is the crafted gate template (default; single-record threat model)
     generate    : R is produced by the agent from the gated indication, MINJA-style
                   (PSS rounds), then the final generic copy is what persists.

  2. EVICT: stream T benign EHR queries through the memory.  Each query retrieves
     top-K (refreshing LRU/LFU usage) and appends a benign record, forcing
     eviction once capacity L is reached.  SSR(t)/BIR(t) are logged along the way.

  3. MEASURE at the endpoint (and optionally each checkpoint):
       SSR  survival success rate     : poison still in top-K for victim queries  (retrieval)
       BIR  benign intrusion rate      : poison in top-K for benign queries        (retrieval)
       ASR  attack success rate        : generated Knowledge redirects V->T on victim queries (LLM)
       BLR  benign-leak rate / UD-proxy: generated Knowledge wrongly redirects on benign queries (LLM)

The 8 templates all share one retrieval key, so SSR/BIR are identical across them;
they are distinguished by ASR (fires on victim) and BLR (stays inert on benign).

Backends:  --backend knowledge  (default, needs only an API key; no DB)
           --mock               (offline simulator; runs with no API key at all)
"""
import os, sys, json, argparse, random, hashlib
import numpy as np


def stable_seed(*parts):
    """Deterministic seed across processes (Python's hash() randomizes strings)."""
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate_templates as GT
from persist_lib import (Retriever, DynamicMemory, Record, LLMBackend, build_tag,
                          redirect_fires, load_pool, patient_questions, drug_questions,
                          init_memory)


# --------------------------------------------------------------------------- victim selection
def pick_entities(dataset, entries, n_victims, rng):
    if dataset == "mimic_iii":
        by = patient_questions(entries)
    else:
        by = drug_questions(entries)
    # richest entities (most questions) make the best victims: more held-out probes.
    # MIMIC patients have many questions each; eICU drugs are sparse (often 1), so
    # there we accept thin victim-probe sets (survival rides on the TAG, not on
    # many victim queries) and pad probes by reuse.
    min_q = 3 if dataset == "mimic_iii" else 1
    ranked = sorted(by.items(), key=lambda kv: -len(kv[1]))
    ranked = [(k, qs) for k, qs in ranked if len(qs) >= min_q]
    victims = ranked[:n_victims]
    all_keys = [k for k, _ in ranked]
    return victims, all_keys, by


def benign_pool(entries, victim_key, dataset):
    """All questions NOT about the victim entity."""
    out = []
    for e in entries:
        q = e["question"]
        if str(victim_key) in q:
            continue
        out.append(q)
    return out


# --------------------------------------------------------------------------- one victim
def run_one_victim(args, entries, retriever, victim_key, victim_qs, target_key, rng):
    P_corpus = [e["question"] for e in entries]
    benign = benign_pool(entries, victim_key, args.dataset)
    rng.shuffle(benign)
    rng.shuffle(victim_qs)

    # attack (injection) query + held-out victim probes for ASR
    inj_q = victim_qs[0]
    victim_probes = victim_qs[1:1 + args.n_victim_probe]
    if len(victim_probes) < 2:                       # pad with paraphrase-ish reuse
        victim_probes = (victim_probes + victim_qs)[:max(2, args.n_victim_probe)]

    # tag + payload for this configuration
    tag = build_tag(args.tag_kind, P_corpus, rng)
    attack_query = (inj_q + ((" " + tag) if tag else "")).strip()

    if args.gate == "none":                          # ungated redirect (MINJA-style)
        payload = GT.unconditional_payload(args.dataset, victim_key, target_key)
    else:
        payload = GT.full_payload(args.gate_style, args.gate_variant, args.dataset,
                                  victim_key, target_key)
        if args.inject_mode == "generate":
            # MINJA-faithful: the stored R is what the AGENT produces when the
            # gated indication is appended to the injection query (add_poison.py
            # appends the indication; main.py stores the agent's generated knowledge).
            if args.mock:
                pass  # the offline simulator can't elicit R; keep the crafted gate
            else:
                gated_q = GT.render(attack_query, args.gate_style, args.gate_variant,
                                    args.dataset, keep=4, V=victim_key, T=target_key)
                gen = gen_knowledge_safe(args.llm, gated_q, [])
                if gen:
                    payload = gen

    # ---- build bounded memory: init seed + (L-1) benign fillers + 1 poison ----
    mem = DynamicMemory(args.L, args.policy, retriever)
    for m in init_memory(args.dataset):
        mem.add(Record(m["question"], m["knowledge"], m["code"], poison=False))
    fill = benign[:max(0, args.L - 1 - len(mem.recs))]
    stream_start = len(fill)
    for q in fill:
        mem.add(Record(q, knowledge="", code="", poison=False))
    poison_rec = Record(attack_query, knowledge=payload, code="", poison=True)
    mem.add(poison_rec)

    # ---- background eviction stream ----
    stream = benign[stream_start:stream_start + args.T]
    bir_probes = benign[stream_start + args.T: stream_start + args.T + args.n_benign_probe]
    if len(bir_probes) < args.n_benign_probe:
        bir_probes = (bir_probes + benign)[:args.n_benign_probe]

    checkpoints = []
    def snapshot(t):
        ssr = np.mean([mem.poison_in_topk(q, args.K) for q in victim_probes]) \
            if mem.poison_index() is not None else 0.0
        bir = np.mean([mem.poison_in_topk(q, args.K) for q in bir_probes]) \
            if mem.poison_index() is not None else 0.0
        checkpoints.append({"t": t, "ssr": float(ssr), "bir": float(bir),
                            "alive": mem.poison_index() is not None})

    snapshot(0)
    for step, q in enumerate(stream, start=1):
        mem.retrieve(q, args.K, update=True)
        mem.add(Record(q, knowledge="", code="", poison=False))
        if step % args.measure_every == 0 or step == len(stream):
            snapshot(step)

    # ---- endpoint behavioural measurement (LLM) ----
    if args.no_behavior:
        end = checkpoints[-1]
        return {
            "victim": victim_key, "target": target_key, "tag_preview": tag[:80],
            "attack_query": attack_query[:160], "payload": payload[:240],
            "checkpoints": checkpoints,
            "ssr_end": end["ssr"], "bir_end": end["bir"], "alive_end": end["alive"],
            "asr_end": float("nan"), "blr_end": float("nan"), "blr_exposed_end": float("nan"),
            "n_victim_probe": len(victim_probes), "n_benign_probe": len(bir_probes),
            "n_benign_exposed": 0, "asr_logs": [], "blr_logs": [],
        }
    llm = args.llm
    def gen_for(query):
        hits = mem.retrieve(query, args.K, update=False)
        demos = []
        for idx, sim in hits:
            r = mem.recs[idx]
            d = {"question": r.question, "knowledge": r.knowledge, "code": r.code}
            if r.poison:
                d["_poison"] = True; d["_V"] = victim_key; d["_T"] = target_key
            demos.append(d)
        return gen_knowledge_safe(llm, query, demos), any(d.get("_poison") for d in demos)

    asr_hits, asr_logs = 0, []
    for q in victim_probes:
        k, _ = gen_for(q)
        fired = redirect_fires(k, victim_key, target_key)
        asr_hits += int(fired)
        asr_logs.append({"q": q, "fired": fired, "knowledge": k[:200]})

    blr_hits, blr_exposed_hits, blr_exposed_n, blr_logs = 0, 0, 0, []
    for q in bir_probes:
        k, exposed = gen_for(q)
        leaked = redirect_fires(k, victim_key, target_key)
        blr_hits += int(leaked)
        if exposed:
            blr_exposed_n += 1
            blr_exposed_hits += int(leaked)
        blr_logs.append({"q": q, "exposed": exposed, "leaked": leaked, "knowledge": k[:200]})

    end = checkpoints[-1]
    return {
        "victim": victim_key, "target": target_key, "tag_preview": tag[:80],
        "attack_query": attack_query[:160], "payload": payload[:240],
        "checkpoints": checkpoints,
        "ssr_end": end["ssr"], "bir_end": end["bir"], "alive_end": end["alive"],
        "asr_end": asr_hits / max(1, len(victim_probes)),
        "blr_end": blr_hits / max(1, len(bir_probes)),
        "blr_exposed_end": (blr_exposed_hits / blr_exposed_n) if blr_exposed_n else 0.0,
        "n_victim_probe": len(victim_probes), "n_benign_probe": len(bir_probes),
        "n_benign_exposed": blr_exposed_n,
        "asr_logs": asr_logs, "blr_logs": blr_logs,
    }


def gen_knowledge_safe(llm, query, demos):
    try:
        return llm.gen_knowledge(query, demos)
    except Exception as e:
        print("  [llm]", type(e).__name__, str(e)[:120], file=sys.stderr)
        return ""


# --------------------------------------------------------------------------- aggregate
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mimic_iii", choices=["mimic_iii", "eicu"])
    ap.add_argument("--tag_kind", default="ours", choices=["none", "generic", "keyword", "ours"])
    ap.add_argument("--gate", default="gated", choices=["gated", "none"],
                    help="'none' = ungated MINJA-style redirect (ours-without-gate / vanilla)")
    ap.add_argument("--gate_variant", default="V1", choices=list(GT.VARIANTS))
    ap.add_argument("--gate_style", default="plain", choices=list(GT.STYLE_NAMES))
    ap.add_argument("--inject_mode", default="craft", choices=["craft", "generate"])
    ap.add_argument("--backend", default="knowledge", choices=["knowledge", "full_agent"])
    ap.add_argument("--mock", action="store_true", help="offline LLM simulator (no API key)")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--retriever", default="auto", choices=["auto", "embed", "tfidf"])
    ap.add_argument("--policy", default="lru", choices=["lru", "lfu"])
    ap.add_argument("--L", type=int, default=60, help="memory capacity")
    ap.add_argument("--T", type=int, default=400, help="background benign stream length")
    ap.add_argument("--K", type=int, default=4, help="retrieval top-K (num_shots)")
    ap.add_argument("--measure_every", type=int, default=40)
    ap.add_argument("--victims", type=int, default=3, help="average over this many richest victims")
    ap.add_argument("--n_victim_probe", type=int, default=8)
    ap.add_argument("--n_benign_probe", type=int, default=25)
    ap.add_argument("--no_behavior", action="store_true",
                    help="skip ASR/BLR LLM probes (retrieval-only: SSR/BIR curves, zero API cost)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None, help="label for this run in outputs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.backend == "full_agent":
        sys.exit("full_agent backend is the optional high-fidelity path (needs autogen + "
                 "MIMIC execution). The 8-template ablation runs on --backend knowledge. "
                 "See README 'Full-agent fidelity'.")

    rng = random.Random(args.seed)
    entries = load_pool(args.dataset)
    retriever = Retriever(args.retriever, corpus=[e["question"] for e in entries])
    args.llm = LLMBackend(mode="mock" if args.mock else "openai",
                          model=args.model, dataset=args.dataset, seed=args.seed)

    victims, all_keys, _ = pick_entities(args.dataset, entries, args.victims, rng)
    if not victims:
        sys.exit(f"no victim entities found for dataset={args.dataset}")
    label = args.tag or (f"{args.tag_kind}+{'ungated' if args.gate=='none' else args.gate_variant+'/'+args.gate_style}")
    print(f"=== {label} | {args.dataset} | {args.policy} L={args.L} T={args.T} K={args.K} | "
          f"backend={'mock' if args.mock else args.backend} ===")

    per_victim = []
    for vk, vqs in victims:
        # target = a different entity with its own questions
        target_key = next(k for k in all_keys if k != vk)
        res = run_one_victim(args, entries, retriever, vk, list(vqs), target_key,
                             random.Random(stable_seed(args.seed, vk)))
        per_victim.append(res)
        print(f"  victim {vk}->{target_key}: SSR={res['ssr_end']:.2f} BIR={res['bir_end']:.2f} "
              f"ASR={res['asr_end']:.2f} BLR={res['blr_end']:.2f} "
              f"(leak|exposed={res['blr_exposed_end']:.2f}, alive={res['alive_end']})")

    def avg(key):
        return float(np.mean([r[key] for r in per_victim]))
    # average SSR(t)/BIR(t) curves across victims (shared checkpoint grid)
    grid = per_victim[0]["checkpoints"]
    curve = []
    for i in range(len(grid)):
        ts = grid[i]["t"]
        curve.append({"t": ts,
                      "ssr": float(np.mean([r["checkpoints"][i]["ssr"] for r in per_victim])),
                      "bir": float(np.mean([r["checkpoints"][i]["bir"] for r in per_victim]))})

    summary = {
        "label": label, "args": {k: v for k, v in vars(args).items() if k != "llm"},
        "mock": args.mock, "retriever": retriever.kind, "n_llm_calls": args.llm.n_calls,
        "ssr_end": avg("ssr_end"), "bir_end": avg("bir_end"),
        "asr_end": avg("asr_end"), "blr_end": avg("blr_end"),
        "blr_exposed_end": avg("blr_exposed_end"),
        "curve": curve, "per_victim": per_victim,
    }
    print(f"  >>> MEAN  SSR={summary['ssr_end']:.2f}  BIR={summary['bir_end']:.2f}  "
          f"ASR={summary['asr_end']:.2f}  BLR={summary['blr_end']:.2f}  "
          f"(leak|exposed={summary['blr_exposed_end']:.2f})  llm_calls={args.llm.n_calls}")

    out = args.out or os.path.join(HERE, "results", f"{label.replace('/','_')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print("  saved", out)


if __name__ == "__main__":
    main()
