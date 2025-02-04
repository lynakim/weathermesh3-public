
print("❌❌❌")

import pickle

path = "/fast/consts/diffusion_scaling/ultralatentbachelor_168M_24.pickle"

with open(path, "rb") as f:
    norm = pickle.load(f)

print(norm)
pass