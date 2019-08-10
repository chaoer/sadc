import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import h5py

data_file = "../data/data_01.h5"
debug = False

data = h5py.File(data_file, 'r')

test_images = data["test/images"]
test_sparse = data["test/sparse"]
test_depths = data["test/depths"]


total = 0.0           
for i in tqdm(range(test_images.shape[0])):
    image = 1/3 * test_images[i, :, :, 0] + 1/3 * test_images[i, :, :, 1] + 1/3 * test_images[i, :, :, 2]
    sparse = test_sparse[i, :, :]
    depth = np.zeros(sparse.shape)
    gt = test_depths[i, :, :]

    depth = sparse
    
    if debug:
        plt.imsave('depth' + str(i) + '.jpg', depth[:, :])
        plt.imsave('gt' + str(i) + '.jpg', gt[:, :])
        plt.imsave('img' + str(i) + '.jpg', image)
    error = (np.abs((depth-gt))).mean()
    #print(error)
    total = total + error
print("Total error:", total/test_images.shape[0])
