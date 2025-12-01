import argparse, csv, numpy as np
from sklearn.cluster import KMeans
import esm

def embed_cdr3s(seqs, model_name="t6_8M", batch=64):
    if model_name == "t6_8M":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    elif model_name == "t12_35M":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    else:
        raise ValueError("model_name must be t6_8M or t12_35M")
    model.eval()
    converter = alphabet.get_batch_converter()
    embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            batch_items = [(str(j), s) for j, s in enumerate(seqs[i:i+batch])]
            _, _, toks = converter(batch_items)
            rep = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
            for k, s in enumerate(seqs[i:i+batch]):
                embs.append(rep[k, 1:1+len(s)].mean(0).cpu().numpy())
    return np.vstack(embs).astype(np.float32)

# torch import after esm to avoid accidental GPU grab (its going to crash if not so just do it)
import torch

def main(inp, outp, k, model_name):
    # load filtered (or unfiltered) CSV
    rows = []
    with open(inp, newline="") as f:
        for row in csv.DictReader(f):
            if "passes" in row and row["passes"] == "N":
                continue
            rows.append(row)
    if not rows:
        raise SystemExit("No rows to cluster (all filtered out?).")

    seqs = [r["cdr3_seq"] for r in rows]
    scores = np.array([float(r["score"]) for r in rows])
    embs = embed_cdr3s(seqs, model_name=model_name, batch=64)

    k = min(k, len(rows))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(embs)
    # pick the highest-scoring member of each cluster
    chosen = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0: continue
        j = idx[np.argmax(scores[idx])]
        chosen.append(rows[j] | {"cluster": c})
    chosen.sort(key=lambda r: -float(r["score"]))

    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chosen[0].keys()))
        w.writeheader(); w.writerows(chosen)
    print(f"Selected {len(chosen)} representatives across {k} clusters → {outp}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="CSV from optimizer or filtered CSV")
    ap.add_argument("--out", required=True, help="Output panel CSV")
    ap.add_argument("--k", type=int, default=12, help="Number of clusters / panel size")
    ap.add_argument("--esm", default="t6_8M", choices=["t6_8M","t12_35M"])
    args = ap.parse_args()
    main(args.inp, args.out, args.k, args.esm)
