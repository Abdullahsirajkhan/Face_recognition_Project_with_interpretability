# Person Identification System Using Facial Recognition with Interpretability Emphasis

## Overview
It is a lightweight interpretable facial recognition system developed using the HOG features and decision tree classifier. The system achieves up to 99% accuracy on our dataset while providing visual and rule-based explanations for each prediction. Designed to address the "black box" problem in AI by applying computational thinking principles.

## Dataset
- Source: Provided by course TA (Dataset was pictures of our classfellows)
- Individuals: 11 (initial description), 12 (EDA section)
- Total Images: 4025
- Format: RGB image frames (~400 per person)
- Variations: Lighting, facial orientation, eye contact, glasses, background
- Issues: Mild class imbalance, inconsistent brightness, one class with different dimensions
- The data set is quite large to upload on GitHub, so I am linking a link to a drive where you can find and download the dataset zipped files for use.
## Download Dataset & Model
Due to size limits, dataset and trained model are stored on Google Drive.

- [Download Dataset_raw](https://drive.google.com/file/d/1WmTC_jJKSypq2EaWL4JQ71XsUtpPHI5d/view?usp=drive_link)
- [Download Dataset_trained](https://drive.google.com/file/d/1WmTC_jJKSypq2EaWL4JQ71XsUtpPHI5d/view?usp=sharing)


## Preprocessing
- Converted images to grayscale
- Resized to 256x256 pixels
- Organized images into class folders
- Tools: OpenCV, NumPy, Pandas, Matplotlib
- Suggested improvements: Face alignment, brightness normalization, data augmentation, class balancing

## Feature Extraction
- Method: Histogram of Oriented Gradients (HOG)
- Steps:
  1. Grayscale conversion
  2. Gradient computation (magnitude & angle)
  3. Divide image into cells, compute histograms
  4. Merge histograms into final feature vectors
- Advantages:
  - Captures facial structure through edges
  - Handles lighting/background variation
  - Produces compact interpretable vectors
- Visualization: HOG feature maps highlight eyes, eyebrows, hair

## Model Training
- Algorithm: Decision Tree Classifier (scikit-learn)
- Train/Test Split: 80% / 20%
- Initial Accuracy: 95%
- Final Accuracy: 99%
- Evaluation: Confusion matrix, classification report
- Challenges:
  - Limited dataset size
  - Variations in shirt patterns/background
  - Class imbalance
  - Lighting, facial angle, expression inconsistencies
- Planned enhancements: SVM, Random Forest (while keeping interpretability), face-only cropping

## Graphical User Interface (GUI)
- Framework: Tkinter
- Features:
  - Upload image → HOG extraction → Prediction → Display label & decision path
  - Live webcam detection & recognition
  - Visualization of HOG features and decision path
- Limitations: Basic aesthetics; can be improved

## Project Pipeline
1. Problem Definition: Focus on interpretability in face recognition
2. Requirements:
   - Python
   - OpenCV
   - scikit-learn
   - matplotlib
   - scikit-image
   - Tkinter
3. Planning & Design: Modular pipeline
4. Implementation: Data preprocessing, HOG extraction, Decision Tree training, decision path extraction, GUI development
5. Testing: Verified model & GUI functionality
6. Deployment: Runs locally (`main.py` for training, `gui.py` for GUI)
7. Maintenance/Future Work: Improve GUI, add normalization/alignment, deploy executable, scalable interpretable models

## Installation & Usage

###
### 2. Install Dependencies

pip install -r requirements.txt

### 3. Prepare Dataset
Default dataset is not included due to size. However a link to dataset in drive is uploaded.

Place dataset in a `data/` folder with the following structure:

data/
 ├── person_01/
 │    ├── img1.jpg
 │    ├── img2.jpg
 │    └── ...
 ├── person_02/
 │    ├── ...
 └── ...

Each folder name should correspond to the class label.

### 4. Train Model (Optional)
If you want to retrain:
python main.py

This will:
- Load and preprocess dataset
- Extract HOG features
- Train Decision Tree classifier
- Save trained model (`model.pkl`)

### 5. Run GUI

python gui.py

GUI Functions:
- Upload Image: Select image → See prediction + HOG + decision path
- Live Camera: Real-time detection & recognition
