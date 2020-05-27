import cv2 as cv
import numpy as np
import os
from plot_images import subplot

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

class ProcessVehicleImage:

    def __init__(self, vehicle_image):
        
        self.vehicle_image = vehicle_image
        self.gray_image = None
        self.sobel_image = None
        self.threshold_image = None
        self.morphed_threshold_image = None
        self.number_plate_detected_image = None
        self.number_plate_image = None
        self.contours = None

    def preprocess(self):
        
        blurred_image = cv.GaussianBlur(self.vehicle_image, (5,5), 0)
        self.gray_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)
        self.sobel_image = cv.Sobel(self.gray_image, cv.CV_8U, 1, 0, ksize = 3)
        _, self.threshold_image = cv.threshold(self.sobel_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        
    def get_contours(self):
        
        kernel_structure = cv.getStructuringElement(shape = cv.MORPH_RECT, ksize = (17, 3))
        self.morphed_threshold_image = self.threshold_image.copy()
        cv.morphologyEx(src = self.threshold_image,
                        op = cv.MORPH_CLOSE,
                        kernel = kernel_structure,
                        dst = self.morphed_threshold_image)
        self.contours, _ = cv.findContours(self.morphed_threshold_image,
                                           mode = cv.RETR_EXTERNAL,
                                           method = cv.CHAIN_APPROX_NONE)

    def detect_plate(self):
        
        self.number_plate_detected_image = self.vehicle_image.copy()
        
        for _, cnt in enumerate(self.contours):
            rectangle_minimum = cv.minAreaRect(cnt)
            if self.check_orientation(rectangle_minimum):
                x, y, w, h = cv.boundingRect(cnt)
                self.number_plate_image = self.number_plate_detected_image[y:y+h, x:x+w]
                if self.check_white_area():
                    rect = self.get_plate_coordinates()
                    if rect:
                        x1, y1, w1, h1 = rect
                        x, y, w, h = x+x1, y+y1, w1, h1
                        self.number_plate_detected_image = cv.rectangle(self.number_plate_detected_image,
                                                                        (x,y), (x+w,y+h), (90,0,255), 2)
                        self.number_plate_image = self.vehicle_image[y:y+h, x:x+w]
    
    def check_white_area(self):
        
        average = np.mean(self.number_plate_image)
        
        if average >= 115:
            return True
        else:
            return False

    def get_plate_coordinates(self):
        
        gray_plate = cv.cvtColor(self.number_plate_image, cv.COLOR_BGR2GRAY)
        _, threshold_plate = cv.threshold(gray_plate, 150, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(threshold_plate.copy(),
                                      mode = cv.RETR_EXTERNAL,
                                      method = cv.CHAIN_APPROX_NONE)
        if contours:
            areas = [cv.contourArea(cnt) for cnt in contours]
            maximum_index = np.argmax(areas)
            maximum_contour = contours[maximum_index]
            maximum_contour_area = areas[maximum_index]
            x, y, w, h = cv.boundingRect(maximum_contour)
            
            if not self.check_ratio(maximum_contour_area, w, h):
                return None
            
            return [x, y, w, h]
        
        else:
            return None

    def check_orientation(self, rectangle_minimum):
        
        (x, y), (w, h), rectangle_angle = rectangle_minimum
        
        if w > h:
            angle = -rectangle_angle
        else:
            angle = 90 + rectangle_angle
        if angle > 15:
            return False
        if h == 0 or w == 0:
            return False
        area = h * w
        if not self.check_ratio(area, w, h):
            return False
        else:
            return True

    def check_ratio(self, area, width, height):
        
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        aspect = 4.7272
        minimum_area = 15 * aspect * 15
        maximum_area = 125 * aspect * 125
        minimum_ratio = 3
        maximum_ratio = 6
        if (area < minimum_area or area > maximum_area) or (ratio < minimum_ratio or ratio > maximum_ratio):
            return False
        return True

    def plot_and_save(self, path):
        
        image_plot = subplot(0.8,([self.vehicle_image,
                                   self.gray_image,
                                   self.sobel_image],
                                  [self.threshold_image,
                                   self.morphed_threshold_image,
                                   self.number_plate_detected_image]))
        cv.imshow("Vehicle Images", image_plot)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(os.path.join(path, 'vehicle_original_image.png'), self.vehicle_image)
        cv.imwrite(os.path.join(path, 'vehicle_gray_image.png'), self.gray_image)
        cv.imwrite(os.path.join(path, 'vehicle_sobel_image.png'), self.sobel_image)
        cv.imwrite(os.path.join(path, 'vehicle_threshold_image.png'), self.threshold_image)
        cv.imwrite(os.path.join(path, 'vehicle_morphed_threshold_image.png'), self.morphed_threshold_image)
        cv.imwrite(os.path.join(path, 'vehicle_number_plate_detected_image.png'), self.number_plate_detected_image)
        cv.imwrite(os.path.join(path, 'vehicle_xplot_image.png'), image_plot)


    def get_number_plate(self):
        
        return self.number_plate_image




class ProcessNumberPlateImage:

    def __init__(self, number_plate_image):
        
        self.number_plate_image = number_plate_image
        self.gray_number_plate = None
        self.threshold_number_plate = None
    
    def preprocess(self):
        
        self.gray_number_plate = self.extract_image_value()
        gray_number_plate_maximum_contrast = self.maximum_contrast()
        height, width = self.gray_number_plate.shape
        imgBlurred = np.zeros((height, width, 1), np.uint8)
        imgBlurred = cv.GaussianBlur(gray_number_plate_maximum_contrast, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
        self.threshold_number_plate = cv.adaptiveThreshold(imgBlurred,
                                                           255.0,
                                                           cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                           cv.THRESH_BINARY_INV,
                                                           ADAPTIVE_THRESH_BLOCK_SIZE,
                                                           ADAPTIVE_THRESH_WEIGHT)
    
    def extract_image_value(self):
        
        height, width, numChannels = self.number_plate_image.shape
        imgHSV = np.zeros((height, width, 3), np.uint8)
        imgHSV = cv.cvtColor(self.number_plate_image, cv.COLOR_BGR2HSV)
        _, _, image_Value = cv.split(imgHSV)
        return image_Value
    
    def maximum_contrast(self):
        
        height, width = self.gray_number_plate.shape
        top_hat = np.zeros((height, width, 1), np.uint8)
        black_hat = np.zeros((height, width, 1), np.uint8)
        kernel_structure = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        top_hat = cv.morphologyEx(self.gray_number_plate, cv.MORPH_TOPHAT, kernel_structure)
        black_hat = cv.morphologyEx(self.gray_number_plate, cv.MORPH_BLACKHAT, kernel_structure)
        gray_plus_top = cv.add(self.gray_number_plate, top_hat)
        gray_plus_top_minus_black = cv.subtract(gray_plus_top, black_hat)
        return gray_plus_top_minus_black

    def plot_and_save(self, path):
        
        image_plot = subplot(1.5,([self.number_plate_image],
                                [self.gray_number_plate],
                                [self.threshold_number_plate]))
        cv.imshow("Numar Plate Images", image_plot)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite(os.path.join(path, 'plate_original_image.png'), self.number_plate_image)
        cv.imwrite(os.path.join(path, 'plate_gray_image.png'), self.gray_number_plate)
        cv.imwrite(os.path.join(path, 'plate_threshold_image.png'), self.threshold_number_plate)
        cv.imwrite(os.path.join(path, 'plate_xplot_image.png'), image_plot)

    def get_number_plate_threshold(self):
        
        return self.threshold_number_plate
    
    
def write_on_image(vehicle_image, registration_number, path):
    cv.putText(vehicle_image, registration_number, (100, 200), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.imshow('Output_Image', vehicle_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite(os.path.join(path, 'output_image.png'), vehicle_image)