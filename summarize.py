"""Collect every runs/*/results.json into one Markdown table and plot learning curves.

  python summarize.py --runs runs            -> runs/summary.md, runs/curves.png
"""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    a = p.parse_args()
    root = Path(a.runs)
    rows = []
    for f in sorted(root.glob("*/results.json")):
        r = json.loads(f.read_text())
        args = r.get("args", {})
        if "test_acc" in r:  # classifier run
            rows.append([f.parent.name, r["mode"], r["arch"], args.get("epochs"), args.get("label_frac", 1.0),
                         f"{r['test_acc']:.2f}", f"{r['train_acc']:.2f}", f"{r['generalization_gap']:.2f}", ""])
        else:  # pretraining run
            rows.append([f.parent.name, f"ssl:{r['method']}", r["arch"], r["epochs"], "-",
                         "", "", "", f"kNN {r['knn_test']:.2f} / LP {r['linear_probe_test']:.2f}"])
    head = ["run", "mode", "arch", "epochs", "label frac", "test acc", "train acc", "gap", "frozen-feature acc"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    md = "\n".join(lines)
    (root / "summary.md").write_text(md + "\n")
    print(md)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping curves)")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for f in sorted(root.glob("*/log.csv")):
        with open(f) as fh:
            data = list(csv.DictReader(fh))
        if not data or "train_acc" not in data[0]:
            continue
        for key, style in [("train_acc", "--"), ("val_acc", "-"), ("test_acc", "-")]:
            pts = [(int(d["epoch"]), float(d[key])) for d in data if d.get(key)]
            if pts:
                ax.plot(*zip(*pts), style, label=f"{f.parent.name} {key.split('_')[0]}")
                plotted = True
    if plotted:
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy (%)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(root / "curves.png", dpi=200)
        print(f"saved {root / 'curves.png'}")


if __name__ == "__main__":
    main()
