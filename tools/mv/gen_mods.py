


text = "\n".join([f"{nix}.npz" for nix in range(0,1728172800,3600*6)])
with open("6hr.txt", "w") as f:
    f.write(text)