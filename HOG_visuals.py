import matplotlib.pyplot as plt

def show_hog_graph(hog_image, predicted_name):
    """
    this function is used to visualize the HOG features extracted from an image. 
    This will help us to undertand model why model predicted a certain name for a given input."""
    plt.imshow(hog_image, cmap='gray')
    plt.title(f"HOG Visualization ({predicted_name})")
    plt.axis('off')
    plt.show()
