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


dsm_dir1 = "raw/DSM_1"
rgb_dir1 = "raw/RGB_1"

dsm_dir2 = "raw/DSM_2"
rgb_dir2 = "raw/RGB_2"

train_pct = .75
n_crops = 10
keep_pct = 0.01
debug = False


#Find and cache all renders, their class, and their mesh
print("Finding renders...")
render_dict = defaultdict(lambda: None)
for root, dirs, files in os.walk(rgb_dir1):
    for file in files:
        if file.endswith(".tif"):
            render_file = os.path.join(root, file)
            
            file_parts = os.path.splitext(file)
            render_class = file_parts[0][4:-4]
            k = render_class

            render_dict[k] = render_file

for root, dirs, files in os.walk(rgb_dir2):
    for file in files:
        if file.endswith(".tif"):
            render_file = os.path.join(root, file)
            
            file_parts = os.path.splitext(file)
            render_class = file_parts[0][4:-4]
            k = render_class

            render_dict[k] = render_file

#Find and cache all DSMs
print("Finding DSMs...")
dsm_list = []
for root, dirs, files in os.walk(dsm_dir1):
    for file in files:
        if file.endswith(".tif"):
            dsm_file = os.path.join(root, file)
            dsm_list.append(dsm_file)

for root, dirs, files in os.walk(dsm_dir2):
    for file in files:
        if file.endswith(".tif"):
            dsm_file = os.path.join(root, file)
            dsm_list.append(dsm_file)

print("Found", len(dsm_list), "DSM files...")

random.shuffle(dsm_list)

#Create train/test partition
train_dsm, test_dsm = train_test_split(dsm_list, train_size=train_pct)
            
#Create testing set
test_depths = np.zeros((n_crops*len(test_dsm), 200, 200))
test_sparse = np.zeros((n_crops*len(test_dsm), 200, 200))
test_images = np.zeros([n_crops*len(test_dsm), 200, 200])
for i, dsm_file in tqdm(enumerate(test_dsm), total=len(test_dsm)):
    with rasterio.open(dsm_file, 'r') as src:
        width = src.width
        height = src.height
    
    file_parts = os.path.splitext(dsm_file)
    k = file_parts[0].split('/')[2][4:]
    render = render_dict[k]
    if render is None:
        k = file_parts[0].split('/')[2][4:-4]
        render = render_dict[k]
    _img = Image.open(render).convert("L")

    for j in range(n_crops):    
        with rasterio.open(dsm_file, 'r') as src:
            matrix = src.transform
            
            row_l = np.random.uniform(0, height-500)
            col_l = np.random.uniform(0, width-500)
            
            row_u = row_l + 500
            col_u = col_l + 500
            
            window = ((row_l, row_u), (col_l, col_u))
            depth = src.read(window=window)
            depth = scipy.misc.imresize(depth[0, :, :], (200, 200))
            test_depths[n_crops*i+j, :, :] = depth
            if debug:
                scipy.misc.imsave('depth' + str(j) + '.jpg', depth[:, :])

            sparse = depth
            k_list = np.random.choice(np.arange(200), size=int(200*200 * (1-keep_pct)))
            l_list = np.random.choice(np.arange(200), size=int(200*200 * (1-keep_pct)))
            for k, l in zip(k_list, l_list):
                sparse[k, l] = -1
            test_sparse[n_crops*i+j, :, :] = sparse
            if debug:
                scipy.misc.imsave('sparse' + str(j) + '.jpg', sparse[:, :])
            
            img = _img.crop(box=(col_l, row_l, col_u, row_u))
            img.thumbnail((200, 200))
            if debug:
                img.save('img' + str(j) + '.jpg', 'JPEG')
            test_images[n_crops*i+j, :, :] = img

total = 0.0           
for i in tqdm(range(len(test_dsm))):
    image = test_images[i, :, :]
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
    labels = KMeans(n_clusters=10).fit_predict(x)
    label_img = np.reshape(labels, [200, 200])
    for label in labels:
        mask = label_img == label
        avg_depth = sparse[mask].sum() / np.count_nonzero(sparse[mask])
        depth[mask] = avg_depth
    if True:
        scipy.misc.imsave('depth' + str(i) + '.jpg', depth[:, :])
        scipy.misc.imsave('gt' + str(i) + '.jpg', gt[:, :])
        plt.imsave('labels' + str(i) + '.jpg', label_img)
        plt.imsave('limg' + str(i) + '.jpg', image)
    error = ((depth-gt)**2).mean()
    print(error)
    total = total + error
print("Total error:", total/len(test_dsm))
