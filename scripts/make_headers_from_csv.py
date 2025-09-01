import numpy as np, json, os
os.makedirs("fpga", exist_ok=True)

def w2d(a,name,fn):
    H,W=a.shape
    with open(fn,"w") as f:
        f.write("#pragma once\n")
        f.write(f"static const float {name}[{H}][{W}] = {{\n")
        for r in range(H):
            row = ", ".join(f"{a[r,c]:.8f}" for c in range(W))
            f.write(f"  {{ {row} }}{',' if r<H-1 else ''}\n")
        f.write("};\n")

def w1d(a,name,fn):
    N=a.shape[0]
    with open(fn,"w") as f:
        f.write("#pragma once\n")
        f.write(f"static const float {name}[{N}] = {{ ")
        f.write(", ".join(f"{v:.8f}" for v in a))
        f.write(" };\n")

# Load CSVs
fc1_w = np.loadtxt("data/fc1_weights.csv",delimiter=",")
fc1_b = np.loadtxt("data/fc1_bias.csv",delimiter=",")
fc2_w = np.loadtxt("data/fc2_weights.csv",delimiter=",")
fc2_b = np.loadtxt("data/fc2_bias.csv",delimiter=",")

w2d(fc1_w,"fc1_weights","fpga/fc1_weights.h")
w1d(fc1_b,"fc1_bias","fpga/fc1_bias.h")
w2d(fc2_w,"fc2_weights","fpga/fc2_weights.h")
w1d(fc2_b,"fc2_bias","fpga/fc2_bias.h")

# Sample input normalization
x = np.load("data/sample_input.npy")[0]
stats = json.load(open("data/normalization.json"))
x_mean, x_std = np.array(stats["x_mean"]), np.array(stats["x_std"])
y_mean, y_std = np.array(stats["y_mean"]), np.array(stats["y_std"])
x_norm = (x-x_mean)/(x_std+1e-8)

with open("fpga/sample_input.h","w") as f:
    f.write("#pragma once\n#define INPUT_SIZE 14\n")
    f.write("static const float sample_input[INPUT_SIZE] = { ")
    f.write(", ".join(f"{v:.8f}" for v in x_norm))
    f.write(" };\n")

with open("fpga/norm.h","w") as f:
    f.write("#pragma once\n#define OUTPUT_SIZE 7\n")
    f.write("static const float y_mean[OUTPUT_SIZE] = { ")
    f.write(", ".join(f"{v:.8f}" for v in y_mean))
    f.write(" };\n")
    f.write("static const float y_std[OUTPUT_SIZE] = { ")
    f.write(", ".join(f"{v:.8f}" for v in y_std))
    f.write(" };\n")

print("Wrote headers into fpga/")
