import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataset_path = r'C:\Users\cz 3\Desktop\ITC_face_recog\dataset'   
save_path = r'C:\Users\cz 3\Desktop' 


os.makedirs(save_path, exist_ok=True)

class_names = []
image_counts = []
image_sizes = []
avg_pixel_intensities = {}

# === Loop through each class folder ===
for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        class_names.append(folder)
        images = os.listdir(folder_path)
        image_counts.append(len(images))

        first_image_path = os.path.join(folder_path, images[0])
        img = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            image_sizes.append(img.shape)
            avg_pixel_intensities[folder] = np.mean(img)

# === Plot 1: Bar Chart - Class Distribution ===
plt.figure(figsize=(12, 6))
sns.barplot(x=class_names, y=image_counts, palette="viridis")
plt.title('Number of Images per Class')
plt.xlabel('Class Name')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'bar_chart_class_distribution.png'), dpi=300)
plt.show()

# === Plot 2: Image Size Consistency ===
unique_sizes = list(set(image_sizes))
print(f"Unique image sizes found: {unique_sizes}")

size_labels = [f"{size[0]}x{size[1]}" for size in unique_sizes]
size_counts = [image_sizes.count(size) for size in unique_sizes]

plt.figure(figsize=(8, 8))
plt.pie(size_counts, labels=size_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Image Size Distribution')
plt.savefig(os.path.join(save_path, 'pie_chart_image_size_distribution.png'), dpi=300)
plt.show()

# === Plot 3: Pixel Intensity Histogram (for one sample image) ===
sample_class = class_names[0]
sample_image_path = os.path.join(dataset_path, sample_class, os.listdir(os.path.join(dataset_path, sample_class))[0])
sample_img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(10, 5))
plt.hist(sample_img.ravel(), bins=256, range=(0, 256), color='purple')
plt.title(f'Pixel Intensity Histogram: {sample_class}')
plt.xlabel('Pixel Intensity (0-255)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(save_path, 'pixel_intensity_histogram.png'), dpi=300)
plt.show()

# === Plot 4: Sample Images Grid ===
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Sample Images from Each Class', fontsize=20)

for i, folder in enumerate(class_names[:12]):
    img_name = os.listdir(os.path.join(dataset_path, folder))[0]
    img_path = os.path.join(dataset_path, folder, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    row, col = divmod(i, 4)
    axes[row, col].imshow(img)
    axes[row, col].set_title(folder)
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(save_path, 'sample_images_grid.png'), dpi=300)
plt.show()

# === Plot 5: Average Pixel Intensity per Class ===
plt.figure(figsize=(12, 6))
sns.barplot(x=list(avg_pixel_intensities.keys()), y=list(avg_pixel_intensities.values()), palette="coolwarm")
plt.title('Average Pixel Intensity per Class')
plt.xlabel('Class Name')
plt.ylabel('Average Pixel Intensity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'average_pixel_intensity_per_class.png'), dpi=300)
plt.show()
