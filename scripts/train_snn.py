import numpy as np, json, torch, torch.nn as nn, torch.optim as optim
import snntorch as snn
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
X = np.load("data/X.npy")
Y = np.load("data/Y.npy")

n = len(X); split = int(0.8 * n)
X_train, Y_train, X_val, Y_val = X[:split], Y[:split], X[split:], Y[split:]

# Normalization
x_mean, x_std = X_train.mean(0), X_train.std(0); x_std[x_std==0]=1
y_mean, y_std = Y_train.mean(0), Y_train.std(0); y_std[y_std==0]=1

np.save("data/sample_input.npy", X[0:1])
np.save("data/sample_output.npy", Y[0:1])
json.dump({"x_mean":x_mean.tolist(),"x_std":x_std.tolist(),
           "y_mean":y_mean.tolist(),"y_std":y_std.tolist()},
          open("data/normalization.json","w"))

def normX(a): return (a-x_mean)/(x_std+1e-8)
def normY(a): return (a-y_mean)/(y_std+1e-8)

# Torch datasets
dl_tr = DataLoader(TensorDataset(torch.tensor(normX(X_train),dtype=torch.float32),
                                 torch.tensor(normY(Y_train),dtype=torch.float32)),
                   batch_size=64, shuffle=True)
dl_va = DataLoader(TensorDataset(torch.tensor(normX(X_val),dtype=torch.float32),
                                 torch.tensor(normY(Y_val),dtype=torch.float32)),
                   batch_size=64)

# Tiny SNN
class TinySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(14,32)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(32,7)
    def forward(self,x,T=8):
        mem = self.lif1.init_leaky()
        acc = 0
        for _ in range(T):
            spk, mem = self.lif1(self.fc1(x), mem)
            acc += self.fc2(spk)
        return acc/T

model = TinySNN()
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for e in range(5):
    model.train(); tot=0
    for xb,yb in dl_tr:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        tot += loss.item()*len(xb)
    print(f"Epoch {e+1}: TrainLoss={tot/len(dl_tr.dataset):.4f}")

torch.save(model.state_dict(), "data/snn_weights.pth")
print("Saved: snn_weights.pth, normalization.json")
