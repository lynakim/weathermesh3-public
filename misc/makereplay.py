import os

def tots(x):
    ls = os.listdir(x)
    ret = []
    for y in ls:
        if y >= "1990" and y < "2022":
            ret.extend([os.path.join(y,z) for z in os.listdir(x+"/"+y) if z.endswith("npz") and int(z.split(".")[0]) % (6*3600) == 0 and os.path.getsize(os.path.join(x,y,z)) in [299013616, 298598896]])
    return ret

ref = "/fast/proc/neo_1_28"
out = "/fast/proc/fakeera5/f000"
full = tots(ref)
full = {k: os.path.join(ref, k) for k in full}

base = "/fast/proc/shortking"
for f in ["f048", "f024"]:
    this = os.path.join(base, f)
    par = tots(this)
    for x in par:
        if x not in full:
            print("uhh", x)
        assert x in full
        if ref in full[x]:
            full[x] = os.path.join(this, x)

print("Total", len(full))
print("orig", len([x for x in full if ref in full[x]]))
for f in ["f048", "f024"]:
    print(f, len([x for x in full if f in full[x]]))

for x in full:
    os.makedirs(out+"/"+x.split("/")[0], exist_ok=True)
    aa = out+"/"+x
    ex = os.path.exists(aa)
    if ex and os.readlink(aa) != full[x]:
        #print("gotta change link!", os.readlink(aa), "to", full[x])
        os.symlink(full[x], aa+".tmp")
        os.rename(aa+".tmp", aa)
    elif not ex:
        #print("new link!")
        os.symlink(full[x], aa)
