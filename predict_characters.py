from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os


model = load_model('ocr.h5')

characters = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
              '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11,
              'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
              'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,
              'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29,
              'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}


def get_key(value): 
    for k, v in characters.items(): 
         if value == v: 
             return k 
             
         
def get_count(path):
    count = 0
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            count += 1
    return count


def predict(path):
    
    count = get_count(path)
    registration_number = ''
    
    for i in range(1, count+1):
        
    
        character_image = image.load_img(os.path.join(path, 'character'+str(i)+'.png'), target_size = (64, 64))
        character_image = image.img_to_array(character_image)
        character_image = np.expand_dims(character_image, axis = 0)
        prediction = model.predict(character_image)
        index = np.where(prediction == 1)[1]
        key = get_key(index[0])
        registration_number = registration_number + key
    
    return registration_number