import os
import cv2
from shutil import copy
from tqdm import tqdm

def get_images_from_folder(folder_path):
    prefix = ['jpg','png']
    imgs = [img for img in os.listdir(folder_path) if img.split('.')[-1] in prefix]
    return imgs

def convert2gray(image_paths, save_path):
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path,0)
        cv2.imwrite(os.path.join(save_path,os.path.basename(img_path)),img)
        name = img_path.replace('.jpg','.txt')
        copy(name, os.path.join(save_path, os.path.basename(name)))

