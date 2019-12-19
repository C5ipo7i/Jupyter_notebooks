# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython


#%%
import numpy as np
import os
import sys
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
get_ipython().run_line_magic('matplotlib', 'inline')

#%% [markdown]
# # Quantatizing Disaster Data
# 
# Dataset is a collection of aerial images of pre and post disaster sites.
#
# ## Extra Datasets
# 
# 1. Extract all the buildings via their polygons and the relevant damage label
# 2. Create another building dataset by upscaling the building images
# 3. Train StyleGAN on this dataset to output damage buildings
# 4. place new damaged building on buildingless images
# 5. train building detector on expanded dataset
#
# ## Goal
# 
# 1. Determine if any buildings exist on a per pixel basis
# 2. Compare the two images and determine the amount of damage done to the structures (if any exist)
# 
# ## Inputs
# 
# Images are 1024x1024 png files
# 
# 1. Given a pair of 1024 images pre and post disaster. 
# 2. output a pair of 1024 images.
# 
# ## Output images
# 
# 1. First image will be 0 or 1 on a per pixel basis determining whether a building is present
# 2. Second image will be 1-4 on a per pixel basis, determining the extent of building damage.

#%%
class DisasterData(object):
    def __init__(self):
        self.folder = '/home/shuza/Code/Data/xView/train'
        self.train_folder = self.folder+'/images'
        self.labels_folder = self.folder+'/labels'
        self.trainset = DisasterData.load_data(self.train_folder)
        self.labels = DisasterData.load_data(self.labels_folder)
        self.trainset_dict = {name:index for index,name in enumerate(self.trainset)}
        self.label_dict = {name:index for index,name in enumerate(self.labels)}
        # Load data pairwise into X and y. Each X should contain two images, pre and post
#         self.X_train = 
#         self.y_train = 
        
    @staticmethod
    def load_data(path):
        data = []
        for file in os.listdir(path):
            if file != '.DS_Store':
                data.append(file)
        return data
        
    def find_label(self,file_name):
        print(file_name)
        label_name = file_name.rstrip('.png')+'.json'
        print('label_name',label_name)
        index = self.label_dict[label_name]
        print('index',index)
        return self.labels[index]
    
    def read_label(self,name):
        label = self.find_label(name)
        print(label)
        label_path = folder+labels_folder+'/'+label
        with open(label_path) as json_file:
            label_im = json.load(json_file)
        print(label_im)
    


#%%
# from PIL import Image
# def save_segmap(segmap_array, file_name):
#     colors = [[  0,   0, 200], # Blue:   Background
#               [  0, 200,   0], # Green:  No Damage
#               [250, 125,   0], # Orange: Minor Damage
#               [250,  25, 150], # Pink:   Major Damage
#               [250,   0,   0]] # Red:    Destroyed
#     colors = np.array(colors).astype(np.uint8)
#     r = Image.fromarray(segmap_array.astype(np.uint8))
#     r.putpalette(colors)
#     r.save(file_name)


#%%
disaster = DisasterData()


#%%
disaster.read_label('guatemala-volcano_00000019_pre_disaster.png')
disaster.read_label('guatemala-volcano_00000019_post_disaster.png')


#%%
disaster.find_label('hurricane-harvey_00000015_pre_disaster.png')
disaster.find_label('hurricane-harvey_00000015_post_disaster.png')

# 'hurricane-harvey_00000015_post_disaster.png'
# 'hurricane-harvey_00000015_pre_disaster.png'
# 'hurricane-harvey_00000015_post_disaster.json'
# 'hurricane-harvey_00000015_pre_disaster.json'


#%%
path = folder+train_folder+'/'+train_images[0]
im = imageio.imread(path)


#%%
label_path = folder+labels_folder+'/'+labels[0]
with open(label_path) as json_file:
    label_im = json.load(json_file)


#%%
print(labels[:5])


#%%



#%%
print(path)
print(im.shape)


#%%
plt.imshow(im)


#%%
label_im


#%%



