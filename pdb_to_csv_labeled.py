# ---- 8/23/25 patch 2
import os, argparse, time, pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.NeighborSearch import NeighborSearch

# ---------- helpers ----------
def chain_to_seq(chain):
    ppb = PPBuilder()
    peptides = ppb.build_peptides(chain)
    if not peptides:
        return None
    return str(max(peptides, key=len).get_sequence())

def is_probable_antibody(seq: str) -> bool:
    if not seq or len(seq) < 80 or len(seq) > 150:
        return False
    s = seq.upper()
    motifs = ("QVQLV","EVQLV","QSVLT","DIVLT","DIQMT","QQKPG","QQRPG")
    if any(m in s for m in motifs):
        return True
    y = s.count("Y")/len(s)
    g = s.count("G")/len(s)
    return (y + g) > 0.22

def pick_heavy_and_antigen(struct):
    model = next(struct.get_models())
    chain_seq = {}
    for ch in model:
        seq = chain_to_seq(ch)
        if seq:
            chain_seq[ch.id] = seq
    if not chain_seq:
        return None, None, {}

    # antibody (VH/VL-ish) and kinda heavy candidates
    ab = {cid for cid, seq in chain_seq.items() if is_probable_antibody(seq)}
    heavy_cands = {
        cid for cid, seq in chain_seq.items()
        if cid in ab and (len(seq) >= 105 or seq.startswith(("QVQL","EVQL")))
    }
    if not heavy_cands:
        heavy_cands = ab.copy()
    if not heavy_cands:
        return None, None, chain_seq

    heavy = max(heavy_cands, key=lambda c: len(chain_seq[c]))

    atoms = [a for a in struct.get_atoms()]
    ns = NeighborSearch(atoms)
    heavy_atoms = [a for a in model.get_atoms()
                   if a.get_parent().get_parent().id == heavy]

    # antigen = non-antibody chain with most 5 A contacts with heavy
    best, best_contacts = None, -1
    for cid, seq in chain_seq.items():
        if cid == heavy or cid in ab:
            continue
        if is_probable_antibody(seq):
            continue
        contacts = 0
        for a in heavy_atoms:
            for n in ns.search(a.coord, 5.0, level="A"):
                if n is a:
                    continue
                if n.get_parent().get_parent().id == cid:
                    contacts += 1
        if contacts > best_contacts:
            best_contacts, best = contacts, cid

    return heavy, best, chain_seq

def crude_cdr3_from_heavy(seq: str) -> str:
    #placeholder. later swap for anarci-based true CDR3 if possible (i tried but anarci isnt compatible with windows -8/23/25)
    tail = seq[-20:]
    return tail if 9 <= len(tail) <= 25 else seq[-15:]

def collect(pdb_dir: str, label: int, tag: str):
    parser = PDBParser(QUIET=True)
    files = [f for f in os.listdir(pdb_dir or "") if f.lower().endswith(".pdb")]
    rows, ok, skipped = [], 0, 0
    t0 = time.time()

    print(f"\n[{tag}] found {len(files)} PDBs in {pdb_dir}")
    for i, fn in enumerate(files, 1):
        path = os.path.join(pdb_dir, fn)
        try:
            struct = parser.get_structure(fn, path)
            heavy, antigen, chain_seq = pick_heavy_and_antigen(struct)
            if not heavy or not antigen:
                skipped += 1
            else:
                heavy_seq = chain_seq.get(heavy, "")
                antigen_seq = chain_seq.get(antigen, "")
                if heavy_seq and antigen_seq:
                    cdr3 = crude_cdr3_from_heavy(heavy_seq)
                    rows.append({"pdb": fn, "antigen_seq": antigen_seq,
                                 "cdr3_seq": cdr3, "label": label})
                    ok += 1
                else:
                    skipped += 1
        except Exception:
            skipped += 1

        # progress ping every 25 files (+ at end)
        if i % 25 == 0 or i == len(files):
            dt = time.time() - t0
            print(f"[{tag}] processed {i}/{len(files)} | ok={ok} skipped={skipped} | {dt:.1f}s elapsed", flush=True)

    return rows

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_dir", required=True, help="folder with positive PDBs")
    ap.add_argument("--neg_dir", required=True, help="folder with negative PDBs")
    ap.add_argument("--out_csv", default="data/antigen_cdr3_pairs.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    pos_rows = collect(args.pos_dir, 1, tag="POS")
    neg_rows = collect(args.neg_dir, 0, tag="NEG")
    rows = pos_rows + neg_rows

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\n wrote {len(df)} labeled pairs to {args.out_csv}")

if __name__ == "__main__":
    main()
