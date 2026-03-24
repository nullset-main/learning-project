# src/features.py
import os
import numpy as np
from tensorflow.keras.preprocessing import image  # type: ignore
import tensorflow as tf

IMG_SIZE = (64, 64)


def preprocess_image(img_path, augment=False):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = np.array(img) / 255.0
    if augment:
        img_array = tf.image.random_flip_left_right(img_array)
        img_array = tf.image.random_brightness(img_array, max_delta=0.1)
    return img_array


def process_folder(input_dir, output_dir, augment=False):
    os.makedirs(output_dir, exist_ok=True)
    for class_name in os.listdir(input_dir):
        class_input_path = os.path.join(input_dir, class_name)
        # Skip non-directories
        if not os.path.isdir(class_input_path):
            continue
        class_output_path = os.path.join(output_dir, class_name)
        os.makedirs(class_output_path, exist_ok=True)
        for img_name in os.listdir(class_input_path):
            # Only allow valid image extensions
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(class_input_path, img_name)
            try:
                processed_img = preprocess_image(img_path, augment)
                save_name = img_name.rsplit(".", 1)[0] + ".npy"
                save_path = os.path.join(class_output_path, save_name)
                np.save(save_path, processed_img)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")
