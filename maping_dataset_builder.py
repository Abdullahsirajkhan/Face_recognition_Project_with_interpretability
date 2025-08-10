import os
from HOG_features_extracting import extract_hog_features

label_map = {
        1: 'Shameer',
        2: 'Shaheer',
        4: 'Saad Ahmed',
        5: 'Jawwad Hussain',
        6: 'Fahad',
        10: 'Kabeer',
        12: 'Abdullah Siraj Khan'
    }

def build_dataset(root_dir):
    X = []
    y = []
    """This function builds a dataset from images in the root_directry
       The images ar expected to be organized in subfolders named by thier labels.
       The function will take the root directory and it will return two lists:
       x: it is a list of feature vectors extracted from the images
       y: it is a list of labels corresponding to the images.
       The labels are derived from the folder names, which should be numeric.
    """
   

    for person_folder in sorted(os.listdir(root_dir)):  # Loop through each person folder in the root directory
        person_path = os.path.join(root_dir, person_folder)  #  Construct the full path to the person's folder
        if not os.path.isdir(person_path): # Checks if the path is a directory
            continue

        try: 
            label = int(person_folder) 
        except ValueError:
            print(f" Skipping folder {person_folder} ( it is not a valid label)") # If the folder name is not a valid integer, skip it
            continue
        person_name = label_map.get(label, 'Unknown')  # Get the person's name from the label map, default to 'Unknown' if not found,
        print(f" Processing label {label} ({person_name})") #   Print the label and name of person

        for img_file in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, img_file)
            try:
                features = extract_hog_features(img_path)  # uses the feature extraction function to get features from the image
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f" Failed to process {img_path}: {e}")

    print(" Dataset built successfully")
    return X, y, label_map
