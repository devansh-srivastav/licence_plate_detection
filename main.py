import cv2 as cv
import os
from process_images import ProcessVehicleImage
from process_images import ProcessNumberPlateImage
from process_images import write_on_image
from segment_characters import segmentation
from predict_characters import predict

image_number = 1
image_extension = '.jpg'
image_path = "car_images/" + str(image_number) + image_extension

root_path = os.getcwd()
results_path = os.path.join(root_path, "results")
target_path = os.path.join(results_path, str(image_number))
character_path = os.path.join(target_path, 'number_plate_characters')
threshold_character_path = os.path.join(target_path, 'number_plate_characters_threshold')

def create_dirs():
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    if not os.path.exists(character_path):
        os.mkdir(character_path)
    if not os.path.exists(threshold_character_path):
        os.mkdir(threshold_character_path)


def main():
    
    vehicle_image = cv.imread(image_path)
    
    create_dirs()
    
    vi = ProcessVehicleImage(vehicle_image)
    vi.preprocess()
    vi.get_contours()
    vi.detect_plate()
    vi.plot_and_save(target_path)
    number_plate_image = vi.get_number_plate()
    
    npi = ProcessNumberPlateImage(number_plate_image)
    npi.preprocess()
    npi.plot_and_save(target_path)
    threshold_number_plate_image = npi.get_number_plate_threshold()
    
    segmentation(number_plate_image.copy(), threshold_number_plate_image.copy(),
                 target_path, character_path, threshold_character_path)
    
    registration_number = predict(threshold_character_path)

    write_on_image(vehicle_image.copy(), registration_number, target_path)
    
    create_file(registration_number)
    
    
def create_file(registration_number):
    file_path = os.path.join(target_path, 'registration_number.txt')
    file = open(file_path, 'w')
    file.write(registration_number)
    file.close()
    
if __name__ == "__main__":
    main()