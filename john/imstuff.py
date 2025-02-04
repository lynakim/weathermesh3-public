import matplotlib.pyplot as plt
import numpy as np
import time



def save_with_metadata(arr,path):
    h, w = arr.shape
    dpi=72
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.axis('off')
    ax.imshow(arr)
    ax.text(10, 40, "Metadata", color="black", fontsize=20,antialiased=False,fontname="monospace")

    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

arr = np.random.rand(1440, 1440)
save_with_metadata(arr,"out.png")