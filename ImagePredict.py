# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 01:37:34 2019

@author: Sathya Bhat
"""
# load and evaluate a saved model
from tensorflow.keras.models import load_model
import glob

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_folder_path = "./FontDetection_Sample"
save_path = "./result.txt"

order = ['Arimo-Regular',
 'DancingScript',
 'FredokaOne',
 'NotoSans',
 'OpenSans',
 'Oswald',
 'PatuaOne',
 'PTSerif',
 'Roboto',
 'Ubuntu']

# load model
model = load_model('small_last4.h5')
# summarize model.
model.summary()
# load dataset

all_images = glob.glob(image_folder_path)
color = 'rgb(0, 255, 0)'
counter = 0

test_datagen = ImageDataGenerator(rescale=1./255)
test_batchsize = 5
test_size = 11

test_generator = test_datagen.flow_from_directory(
        image_folder_path,
        target_size=(224, 224),
        batch_size=test_batchsize,
        class_mode='categorical',
        shuffle=False)

# Get the filenames from the generator
fnames = test_generator.filenames
 
predictions = model.predict_generator(test_generator, steps=test_generator.samples/test_generator.batch_size,verbose=1)
  
file = open(save_path,"a") 

# Show the errors
for i in range(1,predictions.shape[1]+1):
    temp = predictions[i-1:i]
    temp = temp.reshape(predictions.shape[1])
    
    res_index = (-temp).argsort()[:3]
    font_prob = sorted(temp, reverse=True)[0:3]
    

    Res = 'Original label:{}, Prediction-1 :{}, confidence : {:.2f}% Prediction-2 :{}, confidence : {:.2f}% Prediction-3 :{}, confidence : {:.2f}% \n\n '.format(
        fnames[i].split('\\')[0],
        order[res_index[0]],font_prob[0]*100,
        order[res_index[1]],font_prob[1]*100,
        order[res_index[2]],font_prob[2]*100)    
    
    file.write(Res)

file.close()