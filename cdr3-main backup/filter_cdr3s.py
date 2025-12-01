import argparse, csv
from collections import Counter

AA = set("ACDEFGHIKLMNPQRSTVWY")
HYDRO = set("VILFWY")

def has_only_std_aa(s): 
    return set(s) <= AA

def has_nglyc(seq):
    # N-X-[S/T] where X != P
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i+1] != "P" and seq[i+2] in ("S", "T"):
            return True
    return False

def max_hydrophobic_run(seq):
    run = best = 0
    for ch in seq:
        if ch in HYDRO:
            run += 1
            if run > best: best = run
        else:
            run = 0
    return best

def net_charge(seq):
    c = Counter(seq)
    return (c["K"] + c["R"] + 0.1*c["H"]) - (c["D"] + c["E"])

def hard_fail_reason(seq, max_cys, min_len, max_len, max_hydro_run_allowed, max_abs_charge):
    if not has_only_std_aa(seq):           return "non_std_aa"
    if seq.count("C") > max_cys:           return "too_many_cys"
    if has_nglyc(seq):                      return "NXS_T_motif"
    if len(seq) < min_len or len(seq) > max_len:
                                            return "len_out_of_range"
    if max_hydrophobic_run(seq) >= max_hydro_run_allowed:
                                            return "hydrophobic_run>=%d" % max_hydro_run_allowed
    if abs(net_charge(seq)) > max_abs_charge:
                                            return "charge_out_of_range"
    return ""

def main(args):
    rows = []
    with open(args.inp, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            seq = row["cdr3_seq"].strip()
            reason = hard_fail_reason(
                seq,
                max_cys=args.max_cys,
                min_len=args.min_len,
                max_len=args.max_len,
                max_hydro_run_allowed=args.max_hydro_run,
                max_abs_charge=args.max_abs_charge
            )
            # annotate
            row["length"] = len(seq)
            row["charge"] = f"{net_charge(seq):.1f}"
            row["hydro_run"] = max_hydrophobic_run(seq)
            row["has_nglyc"] = has_nglyc(seq)
            row["warn_no_leading_C"] = (seq[:1] != "C")
            row["warn_len"] = (len(seq) < args.min_len or len(seq) > args.max_len)
            row["passes"] = "Y" if reason == "" else "N"
            row["fail_reason"] = reason
            rows.append(row)

    # sort: pass first then by score (desc)
    rows.sort(key=lambda x: (x["passes"] != "Y", -float(x.get("score", 0.0))))

    # If keep_top specified, trim the *passing* set to top K by score
    if args.keep_top and args.keep_top > 0:
        passing = [r for r in rows if r["passes"] == "Y"][:args.keep_top]
        failing = [r for r in rows if r["passes"] != "Y"]
        rows = passing + failing

    # write out
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_pass = sum(r["passes"] == "Y" for r in rows)
    print(f"Wrote {n_pass} passing / {len(rows)} total -> {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="raw optimization CSV")
    ap.add_argument("--out", required=True, help="filtered CSV to write")
    ap.add_argument("--keep_top", type=int, default=0, help="keep top K passing sequences by score (0 = keep all)")

    # optional knobs (matching your current defaults)
    ap.add_argument("--max_cys", type=int, default=2)
    ap.add_argument("--min_len", type=int, default=9)
    ap.add_argument("--max_len", type=int, default=22)
    ap.add_argument("--max_hydro_run", type=int, default=6)
    ap.add_argument("--max_abs_charge", type=float, default=12)

    args = ap.parse_args()
    main(args)
