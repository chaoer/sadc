import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import img_to_graph
from scipy.interpolate import interp2d
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

    #x = []
    #y = []
    #z = []
    #for row in range(200):
    #    for col in range(200):
    #        entry = sparse[row, col]
    #        if entry != 0:
    #            x.append(row)
    #            y.append(col)
    #            z.append(entry)
    #f = interp2d(x, y, z)
    #x = range(0, 200)
    #y = range(0, 200)
    #depth = f(x, y)
    #print(depth.shape)
    x = np.reshape(image, [-1, 1])
    print("Fitting k-means...")
    labels = KMeans(n_clusters=5).fit_predict(x)
    label_img = np.reshape(labels, [200, 200])
    for label in labels:
        mask = label_img == label
        avg_depth = sparse[mask].sum() / np.count_nonzero(sparse[mask])
        depth[mask] = avg_depth
    if debug:
        plt.imsave('depth' + str(i) + '.jpg', depth[:, :])
        plt.imsave('gt' + str(i) + '.jpg', gt[:, :])
        plt.imsave('labels' + str(i) + '.jpg', label_img)
        plt.imsave('limg' + str(i) + '.jpg', image)
    error = (np.abs((depth-gt))).mean()
    print(error)
    total = total + error
print("Total error:", total/test_images.shape[0])
