

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st

# Function to get image files and count classes
def get_image_files(folder_path):
    image_files = []
    class_counts = {}

    for root, dirs, files in os.walk(folder_path):
        # Count images in each subfolder
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            subfolder_image_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if subfolder_image_files:
                image_files.extend(subfolder_image_files)
                class_counts[dir_name] = len(subfolder_image_files)
    
    return image_files, class_counts

# Function to calculate image metrics
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

# Function to plot image metrics
def plot_metrics(folder_path):
    image_files, class_counts = get_image_files(folder_path)
    
    res, blur, noise, hist, sizes, brightness, contrast, sizes_kb = calculate_metrics(image_files)

    figs = []

    # Display total number of images in each class
   

    # Plot distribution of number of images per class
    fig_class_dist, ax_class_dist = plt.subplots(figsize=(8, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    ax_class_dist.bar(classes, counts, color='skyblue')
    ax_class_dist.set_title('Distribution of Number of Images per Class')
    ax_class_dist.set_xlabel('Class (Subfolder)')
    ax_class_dist.set_ylabel('Number of Images')
    ax_class_dist.set_xticklabels(classes, rotation=45, ha='right')
    ax_class_dist.grid(axis='y')
    figs.append(fig_class_dist)

    # Brightness comparison
    fig_brightness, ax_brightness = plt.subplots(figsize=(6, 5))
    ax_brightness.hist(brightness, bins=20, color='green')
    ax_brightness.set_title('Brightness Level Distribution')
    ax_brightness.set_xlabel('Brightness Level')
    ax_brightness.set_ylabel('Frequency')
    ax_brightness.grid(True)
    figs.append(fig_brightness)
    
    # Contrast comparison
    fig_contrast, ax_contrast = plt.subplots(figsize=(6, 5))
    ax_contrast.hist(contrast, bins=20, color='purple')
    ax_contrast.set_title('Contrast Level Distribution')
    ax_contrast.set_xlabel('Contrast Level')
    ax_contrast.set_ylabel('Frequency')
    ax_contrast.grid(True)
    figs.append(fig_contrast)

    # Blur comparison
    fig_blur, ax_blur = plt.subplots(figsize=(6, 5))
    ax_blur.hist(blur, bins=20, color='orange')
    ax_blur.set_title('Blur Measurement Distribution')
    ax_blur.set_xlabel('Variance of Laplacian')
    ax_blur.set_ylabel('Frequency')
    ax_blur.grid(True)
    figs.append(fig_blur)

    # Noise comparison
    fig_noise, ax_noise = plt.subplots(figsize=(6, 5))
    ax_noise.hist(noise, bins=20, color='blue')
    ax_noise.set_title('Noise Level Distribution')
    ax_noise.set_xlabel('Noise Level')
    ax_noise.set_ylabel('Frequency')
    ax_noise.grid(True)
    figs.append(fig_noise)

    # Histogram comparison
    fig_hist, ax_hist = plt.subplots(figsize=(6, 5))
    if hist is not None:
        ax_hist.plot(hist, color='brown')
        ax_hist.set_title('Average Histogram')
        ax_hist.set_xlabel('Pixel Value')
        ax_hist.set_ylabel('Frequency')
    ax_hist.grid(True)
    figs.append(fig_hist)

    # Image size comparison (width and height)
    widths, heights = zip(*sizes)
    fig_size_w, ax_size_w = plt.subplots(figsize=(6, 5))
    bins_w = np.linspace(min(widths), max(widths), 50)
    ax_size_w.hist(widths, bins=bins_w, color='red')
    ax_size_w.set_title('Image Width Distribution')
    ax_size_w.set_xlabel('Width (pixels)')
    ax_size_w.set_ylabel('Frequency')
    ax_size_w.grid(True)
    figs.append(fig_size_w)

    fig_size_h, ax_size_h = plt.subplots(figsize=(6, 5))
    bins_h = np.linspace(min(heights), max(heights), 50)
    ax_size_h.hist(heights, bins=bins_h, color='cyan')
    ax_size_h.set_title('Image Height Distribution')
    ax_size_h.set_xlabel('Height (pixels)')
    ax_size_h.set_ylabel('Frequency')
    ax_size_h.grid(True)
    figs.append(fig_size_h)

    # Image file size comparison (in kilobytes)
    fig_file_size, ax_file_size = plt.subplots(figsize=(6, 5))
    bins_size = np.linspace(min(sizes_kb), max(sizes_kb), 50)
    ax_file_size.hist(sizes_kb, bins=bins_size, color='magenta')
    ax_file_size.set_title('Image File Size Distribution')
    ax_file_size.set_xlabel('File Size (KB)')
    ax_file_size.set_ylabel('Frequency')
    ax_file_size.grid(True)
    figs.append(fig_file_size)

    return figs

# Streamlit app
def main():
    st.title("Data Visualization Tool")

    folder_path = st.text_input("Enter the path for the Folder:")
    
    if st.button("Show Image Metrics"):
        figs = plot_metrics(folder_path)
        for fig in figs:
            st.pyplot(fig)
    
    # Hospital Data Visualization Section
    st.title("Hospital Data Visualization")
    
    # Data from the table
    labels = ['TTQN', 'TTĐT', 'TTĐN']
    good = [11777, 14100, 47712]
    bad = [10329, 100081, 303959]
    total = [22106, 114181, 351671]

    # Calculating percentages
    good_percentage = [g / t * 100 for g, t in zip(good, total)]
    bad_percentage = [b / t * 100 for b, t in zip(bad, total)]

    # Pie chart for each hospital
    st.header("Pie Charts: Percentage of Good and Bad Cases")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, label in enumerate(labels):
        axes[i].pie([good_percentage[i], bad_percentage[i]], labels=['Good', 'Bad'], autopct='%1.1f%%')
        axes[i].set_title(f'Percentage of Good and Bad for {label}')

    st.pyplot(fig)

      # Bar chart for each hospital
    st.header("Bar Chart: Number of Good and Bad Cases in Each Hospital")
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.35
    index = range(len(labels))

    bars1 = ax.bar(index, good, bar_width, label='Good')
    bars2 = ax.bar([p + bar_width for p in index], bad, bar_width, label='Bad')

    ax.set_xlabel('Hospitals')
    ax.set_ylabel('Number of Cases')
    ax.set_title('Number of Good and Bad Cases in Each Hospital')
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels(labels)
    ax.legend()

    st.pyplot(fig)

    # Overall Pie Chart for Good vs Bad
    st.header("Overall Pie Chart: Good vs. Bad")

    # Calculate the total Good and Bad across all hospitals
    total_good = sum(good)
    total_bad = sum(bad)

    # Plot the overall pie chart
    fig_overall, ax_overall = plt.subplots(figsize=(8, 6))
    ax_overall.pie([total_good, total_bad], labels=['Good', 'Bad'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
    ax_overall.set_title('Overall Percentage of Good and Bad Cases')

    st.pyplot(fig_overall)

if __name__ == "__main__":
    main()


