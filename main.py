from image_preprocessing import image_preprocessor  # this imports image processing function from image_preprocessing.py
from HOG_features_extracting import extract_hog_features # this imports the feature extraction function from HOG_features_extracting.py
from maping_dataset_builder import build_dataset  # this imports the dataset builder function from maping_dataset_builder.py
from training_algorithem import train_decision_tree   # this imports the training function from training_algorithem.py
import os
import numpy as np
import joblib

og_dataset_dir = r'C:\Users\cz 3\Desktop\ITC_face_recog\dataset'  # this is the path to the dataset directory
processed_dataset_dir = r'C:\Users\cz 3\Desktop\ITC_face_recog\processed_dataset'  # this is the path to the processed dataset directory

print(" Building dataset to train the model...")  # this prints a message indicating that the dataset building is starting
X, Y, label_map = build_dataset(processed_dataset_dir)  # this calls the dataset builder function to build the dataset from the processed images
print("Dataset building completed.")  # this prints a message indicating that the dataset building is completed"""

print("Training the Decision Tree model...")  # this prints a message indicating that the model training is starting
model = train_decision_tree(X, Y, max_depth=10)  # this calls the training function to train the Decision Tree model on the dataset
print("Model training completed.")  # this prints a message indicating that the model training is completed


joblib.dump(model, 'trained_model.pkl') # this saves the trained model to a file named 'trained_model' in pickel 

print("Model is saved ")
