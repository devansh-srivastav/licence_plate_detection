import numpy as np
import cv2 as cv

def subplot(scale, img_array):
    
    rows = len(img_array)
    columns = len(img_array[0])
    available_rows = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    
    blank_image = np.zeros((height, width, 3), np.uint8)

    if available_rows:
        
        for x in range(rows):
            for y in range(columns):

                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv.resize(img_array[x][y],
                                                (0,0),
                                                None,
                                                scale, scale)
               
                else:
                    img_array[x][y] = cv.resize(img_array[x][y],
                                                (img_array[0][0].shape[1],
                                                 img_array[0][0].shape[0]),
                                                 None,
                                                 scale, scale)
                
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)
                
        horizontal = [blank_image] * rows
                
        for x in range(rows):
            horizontal[x] = np.hstack(img_array[x])
                
        vertical = np.vstack(horizontal)
                
    else:
        
        for x in range(rows):
            
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv.resize(img_array[x],
                                         (0,0),
                                         None,
                                         scale, scale)
            
            else:
                img_array[x] = cv.resize(img_array[x], 
                                         (img_array[0].shape[1],
                                          img_array[0].shape[0]),
                                         None,
                                         scale, scale)
            
            if len(img_array[x].shape) == 2:
                img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)
                
            horizontal = np.hstack(img_array)
            vertical = horizontal
    
    return vertical