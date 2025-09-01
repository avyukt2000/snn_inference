# Minimal training for a tiny SNN policy: 14 -> LIF(32) -> 7
import json, numpy as np, torch, torch.nn as nn, torch.optim as optim
import snntorch as snn
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
X = np.load("data/X.npy")   # shape (N,14)
Y = np.load("data/Y.npy")   # shape (N,7)

# Split
n = len(X); split = int(0.8*n)
Xtr, Ytr = X[:split], Y[:split]
Xva, Yva = X[split:], Y[split:]

# Normalize (train-only stats)
x_mean, x_std = Xtr.mean(0), Xtr.std(0); x_std[x_std==0]=1
y_mean, y_std = Ytr.mean(0), Ytr.std(0); y_std[y_std==0]=1
def nX(a): return (a - x_mean) / (x_std + 1e-8)
def nY(a): return (a - y_mean) / (y_std + 1e-8)

# Tensors & loaders
dl_tr = DataLoader(TensorDataset(torch.tensor(nX(Xtr), dtype=torch.float32),
                                 torch.tensor(nY(Ytr), dtype=torch.float32)),
                   batch_size=64, shuffle=True)
dl_va = DataLoader(TensorDataset(torch.tensor(nX(Xva), dtype=torch.float32),
                                 torch.tensor(nY(Yva), dtype=torch.float32)),
                   batch_size=64, shuffle=False)

# Minimal SNN policy
class TinySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(14, 32)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(32, 7)
    def forward(self, x, T=8):
        mem = self.lif1.init_leaky()
        acc = 0
        for _ in range(T):
            spk, mem = self.lif1(self.fc1(x), mem)
            acc += self.fc2(spk)
        return acc / T  # normalized output (7,)

model = TinySNN()
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Train
for e in range(10):
    model.train(); tot=0
    for xb, yb in dl_tr:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward(); opt.step()
        tot += loss.item()*len(xb)
    # (optional) quick val
    model.eval(); val=0
    with torch.no_grad():
        for xb,yb in dl_va:
            val += loss_fn(model(xb), yb).item()*len(xb)
    print(f"Epoch {e+1}/10  Train {tot/len(dl_tr.dataset):.4f}  Val {val/len(dl_va.dataset):.4f}")

# Save artifacts
torch.save(model.state_dict(), "data/snn_weights.pth")
np.save("data/sample_input.npy", X[:1])
np.save("data/sample_output.npy", Y[:1])
json.dump({"x_mean":x_mean.tolist(),"x_std":x_std.tolist(),
           "y_mean":y_mean.tolist(),"y_std":y_std.tolist()},
          open("data/normalization.json","w"))
print("Saved: data/snn_weights.pth, data/normalization.json, sample_input.npy, sample_output.npy")
