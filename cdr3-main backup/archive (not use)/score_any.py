import sys, numpy as np, torch, torch.nn as nn, esm

AA = "ACDEFGHIKLMNPQRSTVWY" #letter bank

class MLP(nn.Module):
    def __init__(self, d): 
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,512), nn.ReLU(),
                                 nn.Linear(512,128), nn.ReLU(),
                                 nn.Linear(128,1), nn.Sigmoid())
    def forward(self,x): return self.net(x)

def load_scorer(ckpt, d):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mu, sd = ck["mu"], ck["sd"]
    m = MLP(d); m.load_state_dict(ck["state_dict"]); m.eval()
    return m, np.array(mu), np.array(sd)

def embed_seq(model, alphabet, seq):
    conv = alphabet.get_batch_converter()
    with torch.no_grad():
        _,_, toks = conv([("x", seq)])
        rep = model(toks, repr_layers=[model.num_layers])["representations"][model.num_layers]
        return rep[0,1:1+len(seq)].mean(0).cpu().numpy().astype("float32")

if __name__ == "__main__":
    antigen = sys.argv[1].strip()
    cdr3s   = [s.strip() for s in sys.argv[2:]]
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model.eval()

    a = embed_seq(esm_model, alphabet, antigen)
    dim = a.shape[0]*2
    scorer, mu, sd = load_scorer("score_model.pt", dim)
    for s in cdr3s:
        c = embed_seq(esm_model, alphabet, s)
        x = np.concatenate([a, c]).astype("float32")
        x = (x - mu) / (sd + 1e-8)
        with torch.no_grad():
            p = scorer(torch.from_numpy(x).unsqueeze(0)).item()
        print(f"{p:.3f}\t{s}")
