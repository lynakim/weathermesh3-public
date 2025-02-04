import numpy as np
import time 

save_path = "/tmp/a.npy"
save_path = "/fast/b.npy"
a = np.ones((100,1000,1000))

t1 = time.time()
np.save(save_path, a)
#g = np.load(save_path)
t2 = time.time()
print("time:", t2-t1)