# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 21:31:47 2019

@author: Sathya Bhat
"""

from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import glob
import os

def resize2SquareKeepingAspectRation(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = 255*np.ones((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = 255*np.ones((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)


#Path to font
font_path = "./fonts/*.ttf"

#font min size and max size
min_size = 30
max_size = 75

#image size
image_w = 224
image_h = 448

#text postions in image
positions = [(0,image_w/3),(0,0), (0,image_w/2), (0,image_w/4)]

color = 'rgb(0, 0, 0)' # White color
all_fonts = glob.glob(font_path)

train = []
train_labels = []
order = []    

# Choose a font
class_no = 0
for f in all_fonts:
    count=0;
    folderpath = "./dataset/train/" + os.path.basename(f[:-4])
    os.mkdir(folderpath)
    order.append(os.path.basename(f[:-4]))

    for size in range(min_size, max_size, 2):
        for p in positions:
            selected_font = ImageFont.truetype(f, size)
            save = os.path.basename(f[:-4]) + "_" + str(size) + "_" + str(count) + ".png"
                        
            blank_image = 255*np.ones(shape=[image_w, image_h, 3], dtype=np.uint8)
            # Convert to PIL Image
            pil_im = Image.fromarray(blank_image)
            draw = ImageDraw.Draw(pil_im)
            # Draw the text
            draw.text(p, "Hello, World!", fill=color, font=selected_font)
        
            # Save the image
            cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            savepath = folderpath + "/" + save
            
            resized_img = resize2SquareKeepingAspectRation(cv2_im_processed, 224, cv2.INTER_AREA)
            cv2.imwrite(savepath, resized_img)
            
            count = count +1
    
    class_no = class_no+1

#Test dataset
positions = [(0,image_w/3)]

test = []
test_labels = []
    
# Choose a font
class_no = 0
for f in all_fonts:
    count=0;
    folderpath = "./dataset/validation/" + os.path.basename(f[:-4])
    os.mkdir(folderpath)
    for size in range(min_size, max_size, 5):
        for p in positions:
            selected_font = ImageFont.truetype(f, size)
            save = os.path.basename(f[:-4]) + "_" + str(size) + "_" + str(count) + ".png"
                        
            blank_image = 255*np.ones(shape=[image_w, image_h, 3], dtype=np.uint8)
            # Convert to PIL Image
            pil_im = Image.fromarray(blank_image)
            draw = ImageDraw.Draw(pil_im)
            # Draw the text
            draw.text(p, "Hello, World!", fill=color, font=selected_font)
        
            # Save the image
            cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            savepath = folderpath + "/" + save
            
            resized_img = resize2SquareKeepingAspectRation(cv2_im_processed, 224, cv2.INTER_AREA)
            cv2.imwrite(savepath, resized_img)
            
            #test.append(resized_img)
            #test_labels.append(class_no)
            count = count +1
    
    class_no = class_no+1