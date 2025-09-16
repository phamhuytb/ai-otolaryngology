

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

def get_image_files(folder_path):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

def calculate_metrics(image_files):
    resolutions = []
    blur_measurements = []
    noise_levels = []
    average_histograms = []
    sizes = []
    brightness_levels = []
    contrast_levels = []
    file_sizes = []

    for file_path in image_files:
        try:
            image = cv2.imread(file_path)
            if image is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                height, width = image.shape[:2]
                resolution = width * height
                resolutions.append(resolution)

                # Calculate blur measurement
                fm = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_measurements.append(fm)

                # Calculate noise level
                noise_level = np.mean(cv2.blur(gray, (5, 5)) - gray)
                noise_levels.append(noise_level)

                # Store image size
                sizes.append((width, height))

                # Calculate brightness level
                brightness = np.mean(gray)
                brightness_levels.append(brightness)

                # Calculate contrast level
                contrast = gray.std()
                contrast_levels.append(contrast)

                # Calculate histogram
                with Image.open(file_path) as img:
                    grayscale = img.convert("L")
                    histogram = np.array(grayscale.histogram())
                    average_histograms.append(histogram)
                
                # Store file size in kilobytes
                file_size_kb = os.path.getsize(file_path) / 1024
                file_sizes.append(file_size_kb)

            else:
                st.write(f"Could not read image: {file_path}")
        except Exception as e:
            st.write(f"Error processing {file_path}: {e}")

    average_histogram = np.mean(average_histograms, axis=0) if average_histograms else None
    return resolutions, blur_measurements, noise_levels, average_histogram, sizes, brightness_levels, contrast_levels, file_sizes

def plot_comparison(folder1, folder2):
    image_files1 = get_image_files(folder1)
    image_files2 = get_image_files(folder2)
    
    res1, blur1, noise1, hist1, sizes1, brightness1, contrast1, sizes_kb1 = calculate_metrics(image_files1)
    res2, blur2, noise2, hist2, sizes2, brightness2, contrast2, sizes_kb2 = calculate_metrics(image_files2)

    figs = []

    # # Resolution comparison
    # fig_res, ax_res = plt.subplots(figsize=(6, 5))
    # bins = np.linspace(min(min(res1), min(res2)), max(max(res1), max(res2)), 50)
    # ax_res.hist(res1, bins=bins, color='red', alpha=0.7, label='Folder 1')
    # ax_res.hist(res2, bins=bins, color='blue', alpha=0.7, label='Folder 2')
    # ax_res.set_title('Image Resolution Comparison')
    # ax_res.set_xlabel('Resolution (pixels)')
    # ax_res.set_ylabel('Frequency')
    # ax_res.legend()
    # ax_res.grid(True)
    # figs.append(fig_res)
    
    # Brightness comparison
    fig_brightness, ax_brightness = plt.subplots(figsize=(6, 5))
    ax_brightness.hist(brightness1, bins=20, color='red', alpha=0.7, label='Folder 1')
    ax_brightness.hist(brightness2, bins=20, color='blue', alpha=0.7, label='Folder 2')
    ax_brightness.set_title('Brightness Level Comparison')
    ax_brightness.set_xlabel('Brightness Level')
    ax_brightness.set_ylabel('Frequency')
    ax_brightness.legend()
    ax_brightness.grid(True)
    figs.append(fig_brightness)
     # Contrast comparison
    fig_contrast, ax_contrast = plt.subplots(figsize=(6, 5))
    ax_contrast.hist(contrast1, bins=20, color='red', alpha=0.7, label='Folder 1')
    ax_contrast.hist(contrast2, bins=20, color='blue', alpha=0.7, label='Folder 2')
    ax_contrast.set_title('Contrast Level Comparison')
    ax_contrast.set_xlabel('Contrast Level')
    ax_contrast.set_ylabel('Frequency')
    ax_contrast.legend()
    ax_contrast.grid(True)
    figs.append(fig_contrast)

    # Blur comparison
    fig_blur, ax_blur = plt.subplots(figsize=(6, 5))
    ax_blur.hist(blur1, bins=20, color='red', alpha=0.7, label='Folder 1')
    ax_blur.hist(blur2, bins=20, color='blue', alpha=0.7, label='Folder 2')
    ax_blur.set_title('Blur Measurement Comparison')
    ax_blur.set_xlabel('Variance of Laplacian')
    ax_blur.set_ylabel('Frequency')
    ax_blur.legend()
    ax_blur.grid(True)
    figs.append(fig_blur)

    # Noise comparison
    fig_noise, ax_noise = plt.subplots(figsize=(6, 5))
    ax_noise.hist(noise1, bins=20, color='red', alpha=0.7, label='Folder 1')
    ax_noise.hist(noise2, bins=20, color='blue', alpha=0.7, label='Folder 2')
    ax_noise.set_title('Noise Level Comparison')
    ax_noise.set_xlabel('Noise Level')
    ax_noise.set_ylabel('Frequency')
    ax_noise.legend()
    ax_noise.grid(True)
    figs.append(fig_noise)

    # Histogram comparison
    fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
    if hist1 is not None and hist2 is not None:
        ax_hist.plot(hist1, color='red', label='Folder 1')
        ax_hist.plot(hist2, color='blue', label='Folder 2')
        ax_hist.set_title('Average Histogram Comparison')
        ax_hist.set_xlabel('Pixel Value')
        ax_hist.set_ylabel('Frequency')
        ax_hist.legend()
    ax_hist.grid(True)
    figs.append(fig_hist)

    # Image size comparison (width and height)
    widths1, heights1 = zip(*sizes1)
    widths2, heights2 = zip(*sizes2)
    fig_size_w, ax_size_w = plt.subplots(figsize=(6, 5))
    bins_w = np.linspace(min(min(widths1), min(widths2)), max(max(widths1), max(widths2)), 50)
    ax_size_w.hist(widths1, bins=bins_w, color='red', alpha=0.7, label='Folder 1')
    ax_size_w.hist(widths2, bins=bins_w, color='blue', alpha=0.7, label='Folder 2')
    ax_size_w.set_title('Image Width Comparison')
    ax_size_w.set_xlabel('Width (pixels)')
    ax_size_w.set_ylabel('Frequency')
    ax_size_w.legend()
    ax_size_w.grid(True)
    figs.append(fig_size_w)

    fig_size_h, ax_size_h = plt.subplots(figsize=(6, 5))
    bins_h = np.linspace(min(min(heights1), min(heights2)), max(max(heights1), max(heights2)), 50)
    ax_size_h.hist(heights1, bins=bins_h, color='red', alpha=0.7, label='Folder 1')
    ax_size_h.hist(heights2, bins=bins_h, color='blue', alpha=0.7, label='Folder 2')
    ax_size_h.set_title('Image Height Comparison')
    ax_size_h.set_xlabel('Height (pixels)')
    ax_size_h.set_ylabel('Frequency')
    ax_size_h.legend()
    ax_size_h.grid(True)
    figs.append(fig_size_h)

    # Image file size comparison (in kilobytes)
    fig_file_size, ax_file_size = plt.subplots(figsize=(6, 5))
    bins_size = np.linspace(min(min(sizes_kb1), min(sizes_kb2)), max(max(sizes_kb1), max(sizes_kb2)), 50)
    ax_file_size.hist(sizes_kb1, bins=bins_size, color='red', alpha=0.7, label='Folder 1')
    ax_file_size.hist(sizes_kb2, bins=bins_size, color='blue', alpha=0.7, label='Folder 2')
    ax_file_size.set_title('Image File Size Comparison')
    ax_file_size.set_xlabel('File Size (KB)')
    ax_file_size.set_ylabel('Frequency')
    ax_file_size.legend()
    ax_file_size.grid(True)
    figs.append(fig_file_size)

    

   

    return figs

# Streamlit app
def main():
    st.title("Image Comparison Tool")

    folder_path1 = st.text_input("Enter the path for Folder 1:", '/home/huy/huy/DATA/DEEP_MED/DATA/Ear_Nose_Throat/train/')
    folder_path2 = st.text_input("Enter the path for Folder 2:", '/home/huy/huy/DATA/DEEP_MED/DATA/Ear_Nose_Throat/val/')
    
    if st.button("Compare"):
        figs = plot_comparison(folder_path1, folder_path2)
        for fig in figs:
            st.pyplot(fig)

if __name__ == "__main__":
    main()



