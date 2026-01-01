import argparse, random, numpy as np, torch, torch.nn as nn
from tqdm import tqdm
import esm, pandas as pd, os

AA = "ACDEFGHIKLMNPQRSTVWY"

# Matches the MLP architecture from your training code
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def load_scorer(ckpt_path: str, in_dim: int, device="cpu"):
    print(f">>> Loading scorer: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Extracts mu and sd from your training save-state
    mu = torch.from_numpy(ckpt["mu"]).to(device).to(torch.float32)
    sd = torch.from_numpy(ckpt["sd"]).to(device).to(torch.float32)
    model = MLP(in_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, mu, sd

def load_esm(model_name="t12_35M", device="cpu"):
    print(f">>> Loading ESM model: {model_name}")
    if model_name == "t6_8M":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    elif model_name == "t12_35M":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    model.to(device).eval()
    return model, alphabet

def embed_mean(model, alphabet, seqs, device="cpu", batch=64):
    conv = alphabet.get_batch_converter()
    outs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            batch_items = [(str(j), s) for j, s in enumerate(seqs[i:i+batch])]
            _, _, toks = conv(batch_items)
            toks = toks.to(device)
            results = model(toks, repr_layers=[model.num_layers])
            rep = results["representations"][model.num_layers]
            for k, s in enumerate(seqs[i:i+batch]):
                # mean across the sequence length (excluding BOS/EOS)
                outs.append(rep[k, 1:1+len(s)].mean(0))
    return torch.stack(outs)

def mutate(seq, k=1):
    s = list(seq)
    for _ in range(k):
        # Your specific patch: never mutate index 0
        i = random.randrange(1, len(s))
        s[i] = random.choice(AA)
    return "".join(s)

def main():
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="original_pos.csv", help="Your 1027 true pairs")
    parser.add_argument("--output", default="optimized_negatives.csv", help="Where to save 12k lines")
    parser.add_argument("--ckpt", default="score_model.pt")
    parser.add_argument("--esm", default="t12_35M")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--beam", type=int, default=12) # Sets 12 outputs per input
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialization
    esm_model, alphabet = load_esm(args.esm, device)
    
    # Check embedding dimension
    test_emb = embed_mean(esm_model, alphabet, ["CASS"], device)
    in_dim = test_emb.shape[1] * 2 # A + C concatenated
    
    scorer, mu, sd = load_scorer(args.ckpt, in_dim, device)

    # 2. Read Input
    df = pd.read_csv(args.input)
    results_list = []

    print(f"Processing {len(df)} pairs...")

    # 3. Main Loop
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ant_seq = row['antigen_seq']
        cdr_seq = row['cdr3_seq']
        pdb_id = row.get('pdb', 'N/A')

        with torch.no_grad():
            ant_emb = embed_mean(esm_model, alphabet, [ant_seq], device)
            pool = [cdr_seq]
            
            for _ in range(args.steps):
                # Generate variations
                candidates = set(pool)
                for s in pool:
                    for _ in range(3):
                        candidates.add(mutate(s))
                
                candidates = list(candidates)
                cdr_embs = embed_mean(esm_model, alphabet, candidates, device)
                
                # Standardize as per your training code: X = (X - mu) / sd
                # Concatenate [Antigen_repeat, Candidates]
                A_rep = ant_emb.repeat(len(candidates), 1)
                X = torch.cat([A_rep, cdr_embs], dim=1)
                X_scaled = (X - mu) / (sd + 1e-8)
                
                scores = scorer(X_scaled).squeeze(1)
                
                # Select top (beam)
                top_vals, top_idx = torch.topk(scores, min(args.beam, len(candidates)))
                pool = [candidates[i] for i in top_idx]
                pool_scores = top_vals.cpu().numpy()

        # 4. Collect results (12 per input line)
        for i in range(len(pool)):
            results_list.append({
                "pdb": pdb_id,
                "antigen_seq": ant_seq,
                "original_cdr3": cdr_seq,
                "generated_cdr3": pool[i],
                "score": f"{pool_scores[i]:.6f}",
                "label": 0  # These are now your "optimized negatives"
            })

    # 5. Save
    out_df = pd.DataFrame(results_list)
    out_df.to_csv(args.output, index=False)
    print(f"Completed! {len(out_df)} rows written to {args.output}")

if __name__ == "__main__":
    main()