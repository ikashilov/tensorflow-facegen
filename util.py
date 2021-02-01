#!/usr/bin/python
# -*- coding: latin-1 -*-
import os
import numpy as np
import pandas as pd
from PIL import Image
from imageio import imread # as advised https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.imread.html


# as advised https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.imresize.html?highlight=imresize#scipy.misc.imresize
def imresize(arr, size):
    return np.array(Image.fromarray(arr).resize(size=size))


def fetch_lfw_dataset(dirname='./data', use_raw=False, dx=80, dy=80, dimx=45, dimy=45):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        
    images_name = 'lfw' if use_raw else 'lfw-deepfunneled'     
    
    #download dataset
    if not os.path.exists(dirname+'/'+images_name):
        print("images not found, donwloading...")
        os.system(f"wget http://vis-www.cs.umass.edu/lfw/{images_name}.tgz -O tmp.tgz")
        print("extracting...")
        os.system(f"tar xvzf tmp.tgz -C {dirname} && rm tmp.tgz")
        print("done")
        assert os.path.exists(dirname+'/'+images_name) 
    
    #read photos
    print(f"found {images_name} dataset")
    photo_ids = []
    for dirpath, _, filenames in os.walk(dirname+'/'+images_name):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath,fname)
                photo_id = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids.append({'person':person_id,'imagenum':photo_number,'photo_path':fpath})


    #image preprocessing
    print("preprocessing")
    all_photos = pd.DataFrame(photo_ids)['photo_path'].apply(imread) \
                                                       .apply(lambda img: img[dy:-dy,dx:-dx]) \
                                                       .apply(lambda img: imresize(img,[dimx,dimy]))

    data = np.stack(all_photos.values).astype('uint8')
    print("done")
    
    #save processed data
    proc_dirname = dirname+"/processed" 
    if not os.path.exists(proc_dirname):
        os.mkdir(proc_dirname)
        
    print("saving")
    for i, x in enumerate(data):
        Image.fromarray(x).save(f"{proc_dirname}/{i}.png")
    print("done")
    
    return data


def load_lfw_dataset(dirname="./data/processed"):
    if not os.path.exists(dirname):
        raise OSError("Directory {} does not exists".format(dirname))
    
    for root, _, filenames in os.walk(dirname):
        l = [imread(os.path.join(root, fname)) for fname in filenames]
            
    return np.array(l)
