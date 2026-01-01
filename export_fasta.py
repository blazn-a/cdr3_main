import csv, argparse, re
ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", default="CDR3")
args = ap.parse_args()

with open(args.inp) as f, open(args.out, "w") as g:
    r = csv.DictReader(f)
    for i, row in enumerate(r, 1):
        seq = row["cdr3_seq"].strip()
        score = float(row.get("score", 0.0))
        g.write(f">{args.prefix}_{i}|score={score:.3f}\n{seq}\n")
print("wrote", args.out)
