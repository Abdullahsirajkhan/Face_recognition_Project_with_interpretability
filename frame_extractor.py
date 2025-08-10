import cv2
import os
import numpy as np

def extract_frames(video_path, output_dir, num_frames=400):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        print(f"Video only has {total_frames} frames. Reducing to {total_frames} images.")
        num_frames = total_frames

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    current_frame = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_indices:
            filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

            print(f"Saved {filename}")

            if saved_count >= num_frames:
                break

        current_frame += 1

    cap.release()
    print(f"\nExtraction complete: {saved_count} images saved to '{output_dir}'.")

video_path = r"C:\Users\cz 3\Downloads\video.mp4"  
output_dir = r'C:\Users\cz 3\Desktop\dataset\12'

extract_frames(video_path, output_dir)
