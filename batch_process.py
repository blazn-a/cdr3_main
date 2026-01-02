import argparse, random, numpy as np, torch, torch.nn as nn
from tqdm import tqdm
import esm, pandas as pd, os

AA = "ACDEFGHIKLMNPQRSTVWY"

class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def load_esm(model_name="t12_35M", device="cpu"):
    print(f">>> Loading ESM model: {model_name}")
    if model_name == "t6_8M":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    elif model_name == "t12_35M":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    else:
        raise ValueError("model_name must be t6_8M or t12_35M")
    
    # Move model to GPU
    model.to(device).eval() 
    return model, alphabet

def load_scorer(ckpt_path: str, in_dim: int, device="cpu"):
    print(f">>> Loading scorer: {ckpt_path}")
    # FIX: weights_only=False is required for PyTorch 2.6+ when loading checkpoints with numpy data
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Convert mu/sd to tensors on the correct device
    mu = torch.from_numpy(ckpt["mu"]).to(device).to(torch.float32)
    sd = torch.from_numpy(ckpt["sd"]).to(device).to(torch.float32)
    
    model = MLP(in_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, mu, sd

def embed_mean(model, alphabet, seqs, batch=64, device="cpu"):
    """
    Computes mean embeddings.
    Returns: Numpy array (N, dim)
    """
    conv = alphabet.get_batch_converter()
    outs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            batch_items = [(str(j), s) for j, s in enumerate(seqs[i:i+batch])]
            _, _, toks = conv(batch_items)
            
            # Move tokens to the same device as the model
            toks = toks.to(device)
            
            # Forward pass
            res = model(toks, repr_layers=[model.num_layers])
            rep = res["representations"][model.num_layers]
            
            for k, s in enumerate(seqs[i:i+batch]):
                # Slice [1 : len(s)+1] to remove start/end tokens, mean pool, move to CPU numpy
                outs.append(rep[k, 1:1+len(s)].mean(0).detach().cpu().numpy())
                
    return np.vstack(outs).astype(np.float32)

def mutate(seq, k=1):
    s = list(seq)
    L = len(s)
    # Simple mutation: change k residues
    for _ in range(k):
        if L > 1:
            i = random.randrange(0, L) 
            s[i] = random.choice(AA)
    return "".join(s)

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="original_pos.csv", help="Input CSV with antigen_seq and cdr3_seq")
    parser.add_argument("--output", default="optimized_negatives.csv", help="Output CSV")
    parser.add_argument("--ckpt", default="score_model.pt")
    parser.add_argument("--esm", default="t12_35M")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--beam", type=int, default=12) 
    args = parser.parse_args()

    # 1. Initialization
    esm_model, alphabet = load_esm(args.esm, device)
    
    # Check embedding dimension logic
    dummy_emb = embed_mean(esm_model, alphabet, ["CASS"], batch=1, device=device)
    in_dim = dummy_emb.shape[1] * 2 # A + C concatenated
    
    scorer, mu, sd = load_scorer(args.ckpt, in_dim, device)

    # 2. Read Input
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return

    df = pd.read_csv(args.input)
    results_list = []

    print(f"Processing {len(df)} pairs...")

    # 3. Main Loop
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ant_seq = row['antigen_seq']
        cdr_seq = row['cdr3_seq']
        pdb_id = row.get('pdb', 'N/A')

        with torch.no_grad():
            # Get Antigen Embedding (Numpy) -> Convert to Tensor -> GPU
            ant_emb_np = embed_mean(esm_model, alphabet, [ant_seq], batch=1, device=device)
            ant_emb_tensor = torch.from_numpy(ant_emb_np).to(device) # Shape (1, 480)

            pool = [cdr_seq]
            
            for _ in range(args.steps):
                # Generate variations
                candidates = set(pool)
                for s in pool:
                    for _ in range(3):
                        candidates.add(mutate(s))
                
                candidates_list = list(candidates)
                
                # Get Candidate Embeddings (Numpy) -> Convert to Tensor -> GPU
                cdr_embs_np = embed_mean(esm_model, alphabet, candidates_list, batch=64, device=device)
                cdr_embs_tensor = torch.from_numpy(cdr_embs_np).to(device) # Shape (N, 480)
                
                # Concatenate [Antigen_repeat, Candidates]
                A_rep = ant_emb_tensor.repeat(len(candidates_list), 1)
                X = torch.cat([A_rep, cdr_embs_tensor], dim=1)
                
                # Standardize
                X_scaled = (X - mu) / (sd + 1e-8)
                
                # Score
                scores = scorer(X_scaled).squeeze(1) # Shape (N,)
                
                # Select top (beam)
                k = min(args.beam, len(candidates_list))
                top_vals, top_idx = torch.topk(scores, k)
                
                pool = [candidates_list[i] for i in top_idx.cpu().numpy()]

        # 4. Collect results (Exact format: pdb, antigen_seq, cdr3_seq, label)
        for i in range(len(pool)):
            results_list.append({
                "pdb": pdb_id,
                "antigen_seq": ant_seq,
                "cdr3_seq": pool[i], 
                "label": 0  
            })

    # 5. Save
    out_df = pd.DataFrame(results_list)
    out_df.to_csv(args.output, index=False)
    print(f"Completed! {len(out_df)} rows written to {args.output}")

if __name__ == "__main__":
    main()