import matplotlib.pyplot as plt
import numpy as np

def plot_map(attns):
    print(attns.shape, attns.min(), attns.max(), attns.mean())
    b = np.mean(attns, axis=(1,2,3))
    print("yO", b.min(), b.max(), b.mean(), b.shape)
    b.shape = (45, 90)
    plt.imshow(b)
    plt.tight_layout()
    plt.savefig("joan1.png",bbox_inches='tight')

def aa(x):
    plt.imshow(x)
    plt.tight_layout()
    plt.savefig("joan1.png",bbox_inches='tight')
