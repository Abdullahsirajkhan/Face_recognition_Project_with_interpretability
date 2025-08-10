import cv2
import os 
#imports open cv and OS libraries for image processing and file handling
def image_preprocessor(input_dir, output_dir, size=(256,256)): 

    """ The function below takes input directory that contains images, 
        an output directory that is the folder where the processed images
        will store and the Size which is default parameter set to 256by256 pixels."""

    for person in os.listdir(input_dir): # This loops through each person in the input directory
        person_path = os.path.join(input_dir, person)  # joins the input directory with the person folder
        out_path = os.path.join(output_dir, person)  #  check if the output directory exists, if not create it
        os.makedirs(out_path, exist_ok=True)   # this ensures our code doesn't break if the output directory already exists

        for img_file in os.listdir(person_path):  #  This loops through each image file present in each person's folder
           img_path = os.path.join(person_path, img_file)  #  automatically joins the person folder with the image file
           img = cv2.imread(img_path)  # reads the image file using OpenCV
           if img is None:   #  this condition checks if the image is currupted or doesn't exist. 
              continue   #   if the image is currupted, it skips that particular image.
           resized = cv2.resize(img, size)  # tis resize our image to the specified size
           gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) # This converts the resized image into grayscale
           cv2.imwrite(os.path.join(out_path, img_file), gray) # finally, our preprocessed image is saved in its repested path





