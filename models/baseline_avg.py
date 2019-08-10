import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.misc
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import img_to_graph
from scipy.interpolate import interp2d
from itertools import combinations_with_replacement
from collections import defaultdict
import random
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

    if np.count_nonzero(sparse) != 0:
        avg_depth = sparse.sum() / np.count_nonzero(sparse)
    else:
        avg_depth = sparse.sum()
    depth[:, :] = avg_depth
    if debug:
        plt.imsave('depth' + str(i) + '.jpg', depth[:, :])
        plt.imsave('gt' + str(i) + '.jpg', gt[:, :])
        plt.imsave('img' + str(i) + '.jpg', image)
        plt.imsave('sparse' + str(i) + '.jpg', sparse)
    error = (np.abs((depth-gt))).mean()
    #print(error)
    total = total + error
print("Total error:", total/test_images.shape[0])
