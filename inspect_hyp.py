"""Inspect a prediction file from pcl_main.py alongside the gold labels and text.

Usage:
    python inspect_hyp.py testrun.hyp                 # most confident true/false (Q5)
    python inspect_hyp.py testrun.hyp --uncertain     # closest to 0.5 (Q6)
    python inspect_hyp.py vulnerable.hyp --uncertain  # held-out category run (Q7)
    python inspect_hyp.py testrun.hyp -n 10           # show 10 per group instead of 3
"""

import argparse
from lxml import etree

DATA = "/courses/cs159/data/patronize/patronize_full.xml"


def load_hyp(path):
    """Read 'id label confidence' lines into a list of (id, label, conf) tuples."""
    rows = []
    with open(path) as f:
        for line in f:
            ex_id, label, conf = line.split()
            rows.append((ex_id, label, float(conf)))
    return rows


def load_gold(path, wanted_ids):
    """Map example id -> (gold label, category, score, text) for the ids we need."""
    gold = {}
    for _, el in etree.iterparse(path, tag="example"):
        ex_id = el.get("id")
        if ex_id in wanted_ids:
            text = " ".join("".join(el.itertext()).split())
            gold[ex_id] = (el.get("condescension"), el.get("category"),
                           el.get("score"), text)
        el.clear()
    return gold


def show(title, rows, gold):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    for ex_id, pred, conf in rows:
        g_label, category, score, text = gold[ex_id]
        verdict = "CORRECT" if pred == g_label else "WRONG"
        print(f"\n{ex_id}  predicted={pred}  confidence={conf:.6f}  "
              f"gold={g_label} (score={score})  category={category}  [{verdict}]")
        print(f"  {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp_file")
    parser.add_argument("-n", type=int, default=7, help="examples per group")
    parser.add_argument("--uncertain", action="store_true",
                        help="show predictions closest to 0.5 instead of most confident")
    args = parser.parse_args()

    rows = load_hyp(args.hyp_file)

    if args.uncertain:
        groups = [("Least confident predictions (closest to 0.5)",
                   sorted(rows, key=lambda r: abs(r[2] - 0.5))[:args.n])]
    else:
        groups = [
            ("Most confident: predicted CONDESCENDING",
             sorted([r for r in rows if r[1] == "true"], key=lambda r: -r[2])[:args.n]),
            ("Most confident: predicted NOT condescending",
             sorted([r for r in rows if r[1] == "false"], key=lambda r: -r[2])[:args.n]),
        ]

    wanted = {r[0] for _, group in groups for r in group}
    gold = load_gold(DATA, wanted)
    for title, group in groups:
        show(title, group, gold)


if __name__ == "__main__":
    main()