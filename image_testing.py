import cv2
import numpy as np
import joblib
from HOG_features_extracting import extract_hog_features
from maping_dataset_builder import label_map

def predict_single_image(image_path, model_path=r"C:\Users\cz 3\Desktop\ITC_face_recog\trained_model.pkl"):
    """
      This function predicts the label of a single image using a trained model. 
      """

    model = joblib.load(model_path) #This will load th etrained model from the specified path.

    features, _ = extract_hog_features(image_path, visualize=False)
    features = np.array(features).reshape(1, -1)

    predicted_label = model.predict(features)[0] # This uses the trained model to predict the label of the image.
    predicted_name = label_map.get(predicted_label, "Unknown") # This brings the name corresponding to the predicted label from the label map.

    print(f" Prediction: {predicted_name} (Label: {predicted_label})")

    return predicted_label, predicted_name
