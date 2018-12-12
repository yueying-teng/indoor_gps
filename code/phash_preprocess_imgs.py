from imagehash import phash
import os
from PIL import Image
import glob 
from tqdm import tqdm_notebook
import random
import shutil



def grouped_imgs(one_class_img):
    '''
    one_class is the list of image names under one class
    '''

    # dictionary of each image and its corresponding phash
    p2h = {}

    for i in range(len(one_class_img)):
        img = Image.open(one_class_img[i])
        img_hash = phash(img)
        p2h[one_class_img[i][-14: ]] = img_hash
     

    # gather images of the same phash, the unique phash values in this dictionary can be close to each other
    h2ps = {}

    for p, h in enumerate(p2h.items()):
        if h[1] not in h2ps: 
            h2ps[h[1]] = []
        if p not in h2ps[h[1]]:
            h2ps[h[1]].append(p)
       
    # distinct hpash vlaues
    hs = list(h2ps.keys())

    # if the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}
    for i, h1 in enumerate(hs):
        for h2 in hs[: i]:        
            if h1 - h2 <= 8:
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2: 
                    s1, s2 = s2, s1

                h2h[s1] = s2

    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for p, h in p2h.items():
        h = str(h)
        if h in h2h: 
            h = h2h[h]
        p2h[p] = h

    # now all the images in this class can be represented by len(h2ps) number of phashes
    h2ps = {}
    for p, h in p2h.items():
        if h not in h2ps: 
            h2ps[h] = []
        if p not in h2ps[h]: 
            h2ps[h].append(p)

    return h2ps
        


def create_folder_copy_img(train_or_test_folder_dir, pic_data, one_class_name):
    class_folder_dir = os.path.join(train_or_test_folder_dir, one_class_name[-11:])
    
    if not os.path.exists(class_folder_dir):
        os.makedirs(class_folder_dir)

    for i in range(len(pic_data)):
    
        pic_dir = os.path.join(one_class_dir, pic_data[i])
        # copy(src, dst)
        shutil.copy(pic_dir, os.path.join(class_folder_dir, pic_data[i]))




train_folder_dir = '/Users/yueying.teng/Documents/indoor_gps/data/train/' 
test_folder_dir = '/Users/yueying.teng/Documents/indoor_gps/data/test/' 

class_names = glob.glob('/Users/yueying.teng/Documents/indoor_gps/preprocessed_full/class*/')

for i in range(len(class_names)):
    one_class_name = class_names[i]
    one_class_dir = os.path.join(one_class_name, 'output/')
    one_class_img = glob.glob(one_class_dir + 'frame*.jpg')
    h2ps = grouped_imgs(one_class_img)
    print (one_class_name, len(h2ps))
    
    unique_hash = list(h2ps.keys())
    # usable is the list of usabel image names under this class for training and testing 
    usable = []
    for i in unique_hash:
        # use the first image in each group of unique phashes
        usable.append(h2ps[i][0])

    # train_pic and test_pic are two lists with corresponding training and testing image names 
    train_pic = random.sample(usable, int(len(usable)*0.8))
    test_pic = [i for i in usable if i not in train_pic]
    print (len(train_pic), len(test_pic))
    
    # create a subfolder to save images under this class - training 
    create_folder_copy_img(train_folder_dir, train_pic, one_class_name)

    # create a subfolder to save images under this class - testing 
    create_folder_copy_img(test_folder_dir, test_pic, one_class_name)


