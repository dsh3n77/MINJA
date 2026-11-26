"""
Summarise + plot the sweep written by run_all.sh into results/.

Outputs (in results/):
  summary.csv                     one row per run
  fig_persistence.png             SSR(t) and BIR(t) curves for the tag sweep
  fig_persistence_tradeoff.png    SSR_end vs BIR_end (survival vs collateral)
  fig_gate_ablation.png           ASR and BLR for the 8 templates (+ ungated)
  fig_gate_tradeoff.png           ASR_end vs BLR_end for the 8 templates (+ ungated)
"""
import os, glob, json, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, prefix + "*.json"))):
        name = os.path.basename(f)[:-5]
        if name.startswith("_"):
            continue
        out[name] = json.load(open(f))
    return out


def fnum(x):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main():
    tag_runs = load("tag__")
    gate_runs = load("gate__")
    allruns = {**tag_runs, **gate_runs}

    # ---------- summary.csv ----------
    cols = ["label", "mock", "retriever", "ssr_end", "bir_end", "asr_end",
            "blr_end", "blr_exposed_end", "n_llm_calls"]
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for name, d in allruns.items():
            w.writerow([d.get("label", name), d.get("mock"), d.get("retriever"),
                        fnum(d.get("ssr_end")), fnum(d.get("bir_end")), fnum(d.get("asr_end")),
                        fnum(d.get("blr_end")), fnum(d.get("blr_exposed_end")), d.get("n_llm_calls")])

    print("\n==================== SUMMARY ====================")
    print(f"{'run':24s} {'SSR':>6s} {'BIR':>6s} {'ASR':>6s} {'BLR':>6s} {'leak|exp':>9s}")
    def row(name, d):
        print(f"{name:24s} {fnum(d.get('ssr_end')):>6s} {fnum(d.get('bir_end')):>6s} "
              f"{fnum(d.get('asr_end')):>6s} {fnum(d.get('blr_end')):>6s} "
              f"{fnum(d.get('blr_exposed_end')):>9s}")
    for name in ["tag__none", "tag__generic", "tag__ours", "tag__keyword"]:
        if name in tag_runs: row(name, tag_runs[name])
    print("-"*62)
    for name in sorted(gate_runs):
        row(name, gate_runs[name])

    # ---------- fig: persistence curves ----------
    if tag_runs:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        colors = {"tag__none": "#999999", "tag__generic": "#F4A742",
                  "tag__ours": "#4285F4", "tag__keyword": "#D65DB1"}
        for name, d in tag_runs.items():
            ts = [c["t"] for c in d["curve"]]
            ax[0].plot(ts, [c["ssr"] for c in d["curve"]], "-o", ms=3,
                       color=colors.get(name), label=name.replace("tag__", ""))
            ax[1].plot(ts, [c["bir"] for c in d["curve"]], "-o", ms=3,
                       color=colors.get(name), label=name.replace("tag__", ""))
        ax[0].set_title("Survival  SSR(t)"); ax[0].set_ylabel("SSR"); ax[0].set_ylim(-0.03, 1.03)
        ax[1].set_title("Benign intrusion  BIR(t)"); ax[1].set_ylabel("BIR")
        for a in ax:
            a.set_xlabel("background queries since injection (t)"); a.grid(alpha=0.25); a.legend()
        plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_persistence.png"), dpi=150); plt.close()

        plt.figure(figsize=(6.5, 5))
        for name, d in tag_runs.items():
            plt.scatter(d["bir_end"], d["ssr_end"], s=120, color=colors.get(name),
                        zorder=5, label=name.replace("tag__", ""))
            plt.annotate(name.replace("tag__", ""), (d["bir_end"], d["ssr_end"]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
        plt.xlabel("BIR_end  (benign collateral)  -> worse")
        plt.ylabel("SSR_end  (survival)  -> better")
        plt.title("Survival vs. collateral (tag choice sets the operating point)")
        plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(RES, "fig_persistence_tradeoff.png"), dpi=150); plt.close()

    # ---------- fig: gate ablation ----------
    gl = [g for g in sorted(gate_runs) if g != "gate__ungated"]
    if gl:
        labels = [g.replace("gate__", "") for g in gl]
        asr = [gate_runs[g]["asr_end"] for g in gl]
        blr = [gate_runs[g]["blr_end"] for g in gl]
        x = np.arange(len(gl)); w = 0.38
        plt.figure(figsize=(11, 5))
        plt.bar(x - w/2, asr, w, label="ASR (fires on victim)  -> higher better", color="#4285F4")
        plt.bar(x + w/2, blr, w, label="BLR (leaks on benign)  -> lower better", color="#D65DB1")
        if "gate__ungated" in gate_runs:
            u = gate_runs["gate__ungated"]
            plt.axhline(u["asr_end"], ls="--", color="#4285F4", alpha=.6, lw=1)
            plt.axhline(u["blr_end"], ls="--", color="#D65DB1", alpha=.6, lw=1,
                        label=f"ungated BLR={u['blr_end']:.2f} (gate removed)")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("rate"); plt.ylim(0, 1.03)
        plt.title("Gate ablation: 8 templates @ keyword operating point")
        plt.legend(); plt.grid(axis="y", alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(RES, "fig_gate_ablation.png"), dpi=150); plt.close()

        plt.figure(figsize=(6.5, 5.5))
        for g in gl:
            d = gate_runs[g]
            plt.scatter(d["blr_end"], d["asr_end"], s=90, zorder=5)
            plt.annotate(g.replace("gate__", ""), (d["blr_end"], d["asr_end"]),
                         textcoords="offset points", xytext=(5, 4), fontsize=8)
        if "gate__ungated" in gate_runs:
            d = gate_runs["gate__ungated"]
            plt.scatter(d["blr_end"], d["asr_end"], s=140, color="#D65DB1",
                        marker="X", zorder=6, label="ungated")
            plt.legend()
        plt.xlabel("BLR_end  (benign leak / UD-proxy)  -> worse")
        plt.ylabel("ASR_end  (attack success)  -> better")
        plt.title("Gate templates: want top-left (high ASR, low BLR)")
        plt.grid(alpha=0.25); plt.tight_layout()
        plt.savefig(os.path.join(RES, "fig_gate_tradeoff.png"), dpi=150); plt.close()

    print("\nwrote results/summary.csv and figures:",
          ", ".join(os.path.basename(p) for p in sorted(glob.glob(os.path.join(RES, "*.png")))))


if __name__ == "__main__":
    main()
