import os
import rasterio
import numpy as np
from tqdm import tqdm
import scipy.misc
from PIL import Image
from sklearn.model_selection import train_test_split
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
keep_pct = .01
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
'''
for root, dirs, files in os.walk(rgb_dir2):
    for file in files:
        if file.endswith(".tif"):
            render_file = os.path.join(root, file)
            
            file_parts = os.path.splitext(file)
            render_class = file_parts[0][4:-4]
            k = render_class

            render_dict[k] = render_file
'''
#Find and cache all DSMs
print("Finding DSMs...")
dsm_list = []
for root, dirs, files in os.walk(dsm_dir1):
    for file in files:
        if file.endswith(".tif"):
            dsm_file = os.path.join(root, file)
            dsm_list.append(dsm_file)
'''
for root, dirs, files in os.walk(dsm_dir2):
    for file in files:
        if file.endswith(".tif"):
            dsm_file = os.path.join(root, file)
            dsm_list.append(dsm_file)
'''
print("Found", len(dsm_list), "DSM files...")

random.shuffle(dsm_list)

#Create train/test partition
train_dsm, test_dsm = train_test_split(dsm_list, train_size=train_pct)
            
#Create training set
train_depths = np.zeros((n_crops*len(train_dsm), 200, 200))
train_sparse = np.zeros((n_crops*len(train_dsm), 200, 200))
train_images = np.zeros([n_crops*len(train_dsm), 200, 200, 3])
for i, dsm_file in tqdm(enumerate(train_dsm), total=len(train_dsm)):
    with rasterio.open(dsm_file, 'r') as src:
        width = src.width
        height = src.height
    
    file_parts = os.path.splitext(dsm_file)
    k = file_parts[0].split('/')[2][4:]
    render = render_dict[k]
    if render is None:
        k = file_parts[0].split('/')[2][4:-4]
        render = render_dict[k]
    _img = Image.open(render).convert("RGB")

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
            sort_depth = np.zeros(200*200)
            sort_indices = np.argsort(depth.flatten())
            print(sort_indices)
            for pos, index in enumerate(sort_indices):
                sort_depth[index] = pos
            sort_depth = np.reshape(sort_depth, [200, 200])
            print(sort_depth)
            train_depths[n_crops*i+j, :, :] = depth
            if True:
                scipy.misc.imsave('depth' + str(j) + '.jpg', depth[:, :])
                scipy.misc.imsave('sorted_depth' + str(j) + '.jpg', sort_depth[:, :])


            sparse = depth
            k_list = np.random.choice(np.arange(200), size=int(200*200 * (1-keep_pct)))
            l_list = np.random.choice(np.arange(200), size=int(200*200 * (1-keep_pct)))
            for k, l in zip(k_list, l_list):
                sparse[k, l] = -1
            train_sparse[n_crops*i+j, :, :] = sparse
            if debug:
                scipy.misc.imsave('sparse' + str(j) + '.jpg', sparse[:, :])
            
            img = _img.crop(box=(col_l, row_l, col_u, row_u))
            img.thumbnail((200, 200))
            if debug:
                img.save('img' + str(j) + '.jpg', 'JPEG')
            train_images[n_crops*i+j, :, :, :] = img

#Create testing set
test_depths = np.zeros((n_crops*len(test_dsm), 200, 200))
test_sparse = np.zeros((n_crops*len(test_dsm), 200, 200))
test_images = np.zeros([n_crops*len(test_dsm), 200, 200, 3])
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
    _img = Image.open(render).convert("RGB")

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
            test_images[n_crops*i+j, :, :, :] = img
           
#Create HDF5 file and contents
f = h5py.File('data_01.h5','w')
train = f.create_group('train')
tr_images = train.create_dataset('images', data=train_images)
tr_depths = train.create_dataset('depths', data=train_depths)
tr_sprase = train.create_dataset('sparse', data=train_sparse)

test = f.create_group('test')
te_images = test.create_dataset('images', data=test_images)
te_depths = test.create_dataset('depths', data=test_depths)
te_sprase = test.create_dataset('sparse', data=test_sparse)

print("Created dataset with", train_images.shape[0], "training images and", test_images.shape[0], "testing images.")
