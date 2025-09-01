# Load trained SNN and run one forward pass on sample_input.npy
import json, numpy as np, torch, torch.nn as nn
import snntorch as snn

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
        return acc / T  # normalized output

# Load stats & model
stats = json.load(open("data/normalization.json"))
x_mean = np.array(stats["x_mean"]); x_std = np.array(stats["x_std"])
y_mean = np.array(stats["y_mean"]); y_std = np.array(stats["y_std"])

m = TinySNN()
m.load_state_dict(torch.load("data/snn_weights.pth", map_location="cpu"))
m.eval()

# Input -> normalize -> predict -> denormalize
x = np.load("data/sample_input.npy")[0]              # (14,)
xn = (x - x_mean) / (x_std + 1e-8)
y_norm = m(torch.tensor(xn[None,:], dtype=torch.float32)).detach().numpy()[0]
y = y_norm * y_std + y_mean
print("Output (denormalized, 7):", y)
