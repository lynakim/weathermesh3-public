import numpy as np
import matplotlib.pyplot as plt
import os

path = '/fast/proc/era5/extra/'

def view(input_file):
    data = np.load(input_file)['x']
    earth_data = np.load(os.path.join(path, '034_sstk/202009/1600128000.npz'))['x']
    filled = np.where(np.isnan(data), earth_data, data)
    plt.imshow(filled)

def get_size(input_file):
    print(np.load(input_file)['x'].shape)

if __name__ == '__main__':
    input_file = 'tc-maxws/202009/1600128000.npz'
    
    view(os.path.join(path, input_file))
    #get_size(os.path.join(path, input_file))