from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    tx = cuda.threadIdx.x
    ty = cuda.blockIdx.x
    bw = cuda.blockDim.x
    pos = tx + ty * bw
    if pos < x.size:  
        out[pos] = x[pos] + y[pos]

def main():
    n = 100000
    x = np.arange(n).astype(np.float32)
    y = np.arange(n).astype(np.float32)
    out = np.empty_like(x)

    threads_per_block = 128
    blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block

    add_kernel[blocks_per_grid, threads_per_block](x, y, out)
    print("Test completed successfully")

if __name__ == '__main__':
    main()
