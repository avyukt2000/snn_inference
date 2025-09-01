import tensorflow_datasets as tfds
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# Load small 100-episode sample of DROID
ds = tfds.load("droid_100", data_dir="gs://gresearch/robotics", split="train")

X_all, Y_all = [], []

# Scan episodes for the word "move" in the instruction
for episode in ds.shuffle(2000, seed=42).take(50):  # scan first 50
    found = False
    for step in episode["steps"]:
        try:
            instr = step["language_instruction"].numpy().decode("utf-8").lower()
            if "move" in instr:
                found = True
                break
        except Exception:
            pass
    if not found:
        continue

    # Collect states + actions
    for step in episode["steps"]:
        jp = step["observation"]["joint_position"].numpy()     # 7
        cp = step["observation"]["cartesian_position"].numpy() # 6
        gp = step["observation"]["gripper_position"].numpy()   # 1
        X_all.append(np.concatenate([jp, cp, gp]))             # shape (14,)
        Y_all.append(step["action"].numpy())                   # shape (7,)

X, Y = np.array(X_all), np.array(Y_all)
np.save("data/X.npy", X)
np.save("data/Y.npy", Y)
print("Saved:", X.shape, Y.shape)
