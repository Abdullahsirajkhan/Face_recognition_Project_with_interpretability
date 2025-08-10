import cv2
import numpy as np
import joblib
from skimage.feature import hog
import tkinter as tk
from tkinter import filedialog, messagebox
from maping_dataset_builder import label_map

# Load trained model
model = joblib.load(r'C:\Users\cz 3\Desktop\ITC_face_recog\trained_model.pkl')

# Extract HOG features
def extract_hog_features(image, size=(256, 256)):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_resized = cv2.resize(image_gray, size)
    features, _ = hog(image_resized, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualize=True, feature_vector=True)
    return features

# Predict from face region
def predict_face(face_img):
    features = extract_hog_features(face_img)
    features = np.array(features).reshape(1, -1)
    label = model.predict(features)[0]
    name = label_map.get(label, "Unknown")
    return name

# Single Image Prediction
def predict_from_image():
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if not image_path:
        return

    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load image.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_roi = image[y:y + h, x:x + w]
        name = predict_face(face_roi)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow("Image Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Live Camera Prediction
def live_camera():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            name = predict_face(face_roi)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Live Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Setup
root = tk.Tk()
root.title("Face Recognition System")

# Full Screen and Background
root.attributes('-fullscreen', True)
root.configure(bg="#1e1e1e")

# Style for Buttons
btn_style = {
    'font': ("Arial", 16),
    'width': 25,
    'bg': '#007ACC',
    'fg': 'white',
    'activebackground': '#005A9E',
    'relief': 'raised',
    'bd': 4
}

# Title Label
tk.Label(root, text="Face Recognition System", font=("Arial", 32), bg="#1e1e1e", fg="white").pack(pady=50)

# Buttons
tk.Button(root, text="Live Face Detection", command=live_camera, **btn_style).pack(pady=20)
tk.Button(root, text="Single Image Prediction", command=predict_from_image, **btn_style).pack(pady=20)
tk.Button(root, text="Exit", command=root.destroy, **btn_style).pack(pady=20)

# Escape key to exit full screen
root.bind("<Escape>", lambda e: root.attributes('-fullscreen', False))

root.mainloop()
