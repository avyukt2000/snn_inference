import torch, numpy as np
from train_snn import TinySNN  # or redefine same class here

state = torch.load("data/snn_weights.pth", map_location="cpu")

np.savetxt("data/fc1_weights.csv", state["fc1.weight"].numpy(), delimiter=",")
np.savetxt("data/fc1_bias.csv",    state["fc1.bias"].numpy(),   delimiter=",")
np.savetxt("data/fc2_weights.csv", state["fc2.weight"].numpy(), delimiter=",")
np.savetxt("data/fc2_bias.csv",    state["fc2.bias"].numpy(),   delimiter=",")
print("Exported CSV files to data/")
