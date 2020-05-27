import cv2 as cv
import numpy as np
import os


def segmentation(number_plate_image, threshold_number_plate_image, target_path, character_path, threshold_character_path):
    
    list_of_characters = []
    list_of_characters_threshold = []
     
    contours, _ = cv.findContours(threshold_number_plate_image,
                                  cv.RETR_EXTERNAL,
                                  cv.CHAIN_APPROX_SIMPLE)    
    
    sorted_contours = sorted(contours, key=lambda ctr: cv.boundingRect(ctr)[0])
    
    plate_height = number_plate_image.shape[0]
    character_count = 0
    
    for i, ctr in enumerate(sorted_contours):

        x, y, w, h = cv.boundingRect(ctr)
        
        if h > w and h < 4 * w and h > plate_height / 3:
            
            character_count += 1
            roi = number_plate_image [y:y+h, x:x+w]
            roi_threshold = np.invert(threshold_number_plate_image [y:y+h, x:x+w])
            list_of_characters.append(roi.copy())
            list_of_characters_threshold.append(roi_threshold.copy())
            cv.imwrite(os.path.join(character_path, 'character'+str(character_count)+'.png'), roi)
            cv.imwrite(os.path.join(threshold_character_path, 'character'+str(character_count)+'.png'), roi_threshold)
            cv.rectangle(number_plate_image,(x,y),( x + w, y + h ),(90,0,255),2)
            
    for i in list_of_characters:
        cv.imshow('Character', i)
        cv.waitKey(0)
        cv.destroyAllWindows()
        
    for i in list_of_characters_threshold:
        cv.imshow('Character', i)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    cv.imshow('Segmented Number Plate',number_plate_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite(os.path.join(target_path, 'segmented_number_plate.png'), number_plate_image)