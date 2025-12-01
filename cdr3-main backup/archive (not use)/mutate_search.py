import numpy as np, torch, torch.nn as nn, random

PAIRS = "data/pair_embeddings.npy"
MODEL = "score_model.pt"
AA = list("ACDEFGHIKLMNPQRSTVWY")

pairs = np.load(PAIRS); A, C = pairs[:,0,:], pairs[:,1,:]
N, D = A.shape
antigen = A[0]  # pick first antigen
base_cdr3 = C[0].copy()

class MLP(nn.Module):
    def __init__(self, in_dim): 
        super().__init__(); 
        self.net = nn.Sequential(nn.Linear(in_dim,512), nn.ReLU(),
                                 nn.Linear(512,128), nn.ReLU(),
                                 nn.Linear(128,1), nn.Sigmoid())
    def forward(self,x): return self.net(x)

model = MLP(in_dim=2*D)
model.load_state_dict(torch.load(MODEL, map_location="cpu"))
model.eval()

def score(ant, cdr):
    x = np.concatenate([ant, cdr])[None, :].astype(np.float32)
    with torch.no_grad(): return float(model(torch.from_numpy(x)).item())

best = base_cdr3; best_s = score(antigen, best)
print("start score:", best_s)

for it in range(100):  # iterations
    cand = best.copy()
    # simple gaussian noise in embedding space to mimic a sequence tweak
    cand += np.random.normal(scale=0.05, size=cand.shape)
    s = score(antigen, cand)
    if s > best_s:
        best, best_s = cand, s
        print(f"iter {it:03d} | improved -> {best_s:.3f}")

print("best score:", best_s)
