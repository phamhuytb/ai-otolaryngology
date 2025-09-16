



import streamlit as st
import os
from PIL import Image, ImageStat
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

def analyze_image_dataset(dataset_path):
    image_info = defaultdict(list)
    total_size = 0
    class_distribution = defaultdict(int)
    unique_sizes = defaultdict(int)

    def process_folder(folder_path):
        nonlocal total_size
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    file_path = os.path.join(root, file)
                    label = os.path.relpath(root, dataset_path)
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                            unique_sizes[(width, height)] += 1
                            aspect_ratio = width / height
                            format = img.format
                            channels = 3 if img.mode == 'RGB' else (1 if img.mode == 'L' else len(img.getbands()))
                            img_array = np.array(img)
                            contrast = img_array.std()
                            brightness = ImageStat.Stat(img.convert('L')).mean[0]  # Tính độ sáng
                            file_size = os.path.getsize(file_path)
                            total_size += file_size
                            class_distribution[label] += 1

                            image_info['sizes'].append((width, height))
                            image_info['aspect_ratios'].append(aspect_ratio)
                            image_info['formats'].append(format)
                            image_info['channels'].append(channels)
                            image_info['contrasts'].append(contrast)
                            image_info['brightness'].append(brightness)

                    except Exception as e:
                        st.error(f"Error processing {file_path}: {e}")

    process_folder(dataset_path)
    return image_info, total_size, class_distribution, unique_sizes

def gather_analysis_results(image_info, total_size, class_distribution):
    results = {
        "sizes": image_info['sizes'],
        "size_min": min(image_info['sizes']),
        "size_max": max(image_info['sizes']),
        "aspect_ratio_min": min(image_info['aspect_ratios']),  # Tỷ lệ khung hình nhỏ nhất
        "aspect_ratio_max": max(image_info['aspect_ratios']),  # Tỷ lệ khung hình lớn nhất
        "aspect_ratio_avg": np.mean(image_info['aspect_ratios']),  # Tỷ lệ khung hình trung bình
        "formats": dict(zip(*np.unique(image_info['formats'], return_counts=True))),
        "channels": dict(zip(*np.unique(image_info['channels'], return_counts=True))),
        "contrast_min": min(image_info['contrasts']),  # Độ tương phản nhỏ nhất
        "contrast_max": max(image_info['contrasts']),  # Độ tương phản lớn nhất
        "contrast_avg": np.mean(image_info['contrasts']),  # Độ tương phản trung bình
        "brightness_min": min(image_info['brightness']),  # Độ sáng nhỏ nhất
        "brightness_max": max(image_info['brightness']),  # Độ sáng lớn nhất
        "brightness_avg": np.mean(image_info['brightness']),  # Độ sáng trung bình
        "class_distribution": dict(class_distribution),
        "total_size_mb": total_size / (1024 * 1024),
        "num_classes": len(class_distribution),
        "min_class": min(class_distribution, key=class_distribution.get),
        "max_class": max(class_distribution, key=class_distribution.get),
        "min_class_count": min(class_distribution.values()),
        "max_class_count": max(class_distribution.values())
    }
    return results

def plot_analysis_results(results, unique_sizes):
    # Plotting Class Distribution
    class_distribution = results['class_distribution']
    fig, ax = plt.subplots()
    ax.bar(class_distribution.keys(), class_distribution.values())
    plt.title('Class Distribution')
    plt.xticks(rotation=90)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    st.pyplot(fig)

    # Plotting Aspect Ratios
    aspect_ratios = results['aspect_ratio_min'], results['aspect_ratio_max'], results['aspect_ratio_avg']
    fig, ax = plt.subplots()
    ax.bar(['Min AR', 'Max AR', 'Avg AR'], aspect_ratios)
    plt.title('Aspect Ratios')
    st.pyplot(fig)

    # Plotting Image Contrast
    contrasts = results['contrast_min'], results['contrast_max'], results['contrast_avg']
    fig, ax = plt.subplots()
    ax.bar(['Min Contrast', 'Max Contrast', 'Avg Contrast'], contrasts)
    plt.title('Image Contrast')
    st.pyplot(fig)

    # Plotting Image Brightness
    brightness = results['brightness_min'], results['brightness_max'], results['brightness_avg']
    fig, ax = plt.subplots()
    ax.bar(['Min Brightness', 'Max Brightness', 'Avg Brightness'], brightness)
    plt.title('Image Brightness')
    st.pyplot(fig)

    # Plotting Number of Images for Each Size
    size_keys = [f"{w}x{h}" for w, h in unique_sizes.keys()]
    size_values = unique_sizes.values()
    fig, ax = plt.subplots()
    ax.bar(size_keys, size_values)
    plt.title('Number of Images for Each Size')
    plt.xticks(rotation=90)
    plt.xlabel('Image Size (WxH)')
    plt.ylabel('Number of Images')
    st.pyplot(fig)

    # Histogram of image sizes
    widths, heights = zip(*results['sizes'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.hist(widths, bins=30, color='blue', edgecolor='blacktes')
    ax1.set_title('Distribution of Image Widths')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Frequency')

    ax2.hist(heights, bins=30, color='green', edgecolor='black')
    ax2.set_title('Distribution of Image Heights')
    ax2.set_xlabel('Height')
    ax2.set_ylabel('Frequency')

    st.pyplot(fig)

def find_most_common_size(unique_sizes):
    most_common_size = max(unique_sizes, key=unique_sizes.get)
    return most_common_size, unique_sizes[most_common_size]

def show_sample_images(dataset_path, num_samples=5):
    sample_images = defaultdict(list)

    def gather_samples(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    label = os.path.relpath(root, dataset_path)
                    sample_images[label].append(os.path.join(root, file))
                    if len(sample_images[label]) >= num_samples:
                        continue

    gather_samples(dataset_path)
    st.write(f"Displaying {num_samples} samples from each class:")
    for label, images in sample_images.items():
        st.write(f"Class: {label}")
        cols = st.columns(num_samples)
        for i, file_path in enumerate(images):
            if i < num_samples:
                img = Image.open(file_path)
                with cols[i % num_samples]:
                    st.image(img, caption=os.path.basename(file_path), use_column_width=True)



def plot_bar_and_pie_chart():
    # Data from the image provided
    data = {
        'Category': ['TTCL', 'TTNT', 'TTSG', 'TTHN', 'PK', 'TTQN'],
        'Quantity': [10839, 136949, 10391, 5156, 16914, 111178]
    }

    df = pd.DataFrame(data)

    # Vẽ biểu đồ cột
    fig, ax = plt.subplots()
    ax.bar(df['Category'], df['Quantity'], color='skyblue')
    ax.set_xlabel('Category')
    ax.set_ylabel('Quantity')
    ax.set_title('Quantity by Category')
    st.pyplot(fig)

    # Vẽ biểu đồ tròn
    fig, ax = plt.subplots()
    ax.pie(df['Quantity'], labels=df['Category'], autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    ax.set_title('Percentage by Category')
    st.pyplot(fig)

def plot_data_clean_pie_chart():
    # Dữ liệu của bạn
    total_quantity = 281036
    data_clean = 94917

    # Tính phần trăm
    clean_percentage = data_clean / total_quantity * 100
    other_percentage = (total_quantity - data_clean) / total_quantity * 100

    # Dữ liệu cho biểu đồ tròn
    labels = ['Data Clean', 'Data_raw']
    sizes = [clean_percentage, other_percentage]
    colors = ['lightcoral', 'lightskyblue']

    # Vẽ biểu đồ tròn
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    ax.set_title(' Data Clean vs Data_raw')
    st.pyplot(fig)


def plot_data_lable_pie_chart():
    # Dữ liệu của bạn
    data_clean = 94917
    data_lable = 4314

    # Tính phần trăm
    clean_percentage = data_lable / data_clean * 100
    other_percentage = (data_clean - data_lable) / data_clean * 100

    # Dữ liệu cho biểu đồ tròn
    labels = ['Data lable', 'Data_raw']
    sizes = [clean_percentage, other_percentage]
    colors = ['lightcoral', 'lightskyblue']

    # Vẽ biểu đồ tròn
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    ax.set_title(' Data Clean vs Data_lable')
    st.pyplot(fig)

def plot_data_lable():
    # Data from the provided image
    categories = ['Tai (Ear)', 'Mũi (Nose)', 'Họng (Throat)']
    cases = [1604, 1225, 1485]
    total_cases = sum(cases)

    # Calculating the percentage of each category
    percentages = [x / total_cases * 100 for x in cases]

    # Creating the pie chart
    fig, ax = plt.subplots()
    ax.pie(percentages, labels=categories, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title('Percentage of ENT Lable')
    st.pyplot(fig)
def plot_data_lable_1():
    # Data from the provided categories and details
    ear_conditions = {'Viêm tai giữa ứ dịch': 177, 'Viêm tai giữa mãn tính': 333, 'Nấm ống tai ngoài': 247,
                    'Viêm ống tai ngoài cấp': 179, 'Viêm tai giữa cấp': 317, 'Tai bình thường': 351}
    nose_conditions = {'Viêm mũi dị ứng': 278, 'Viêm mũi xoang cấp': 189, 'Polyp mũi': 211,
                    'Vẹo vách ngăn': 227, 'Mũi bình thường': 320}
    throat_conditions = {'U vòm lành tính caVA': 293, 'Viêm thanh quản': 274, 'U lành tính': 349,
                        'Thanh quản bình thường': 324, 'Vòm bình thường': 245}

    # Calculating percentages
    total_ear = sum(ear_conditions.values())
    total_nose = sum(nose_conditions.values())
    total_throat = sum(throat_conditions.values())

    ear_percentages = {k: v / total_ear * 100 for k, v in ear_conditions.items()}
    nose_percentages = {k: v / total_nose * 100 for k, v in nose_conditions.items()}
    throat_percentages = {k: v / total_throat * 100 for k, v in throat_conditions.items()}

    # Creating pie charts
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Ear conditions pie chart
    axs[0].pie(ear_percentages.values(), labels=ear_percentages.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
    axs[0].set_title('Ear Conditions Percentage')

    # Nose conditions pie chart
    axs[1].pie(nose_percentages.values(), labels=nose_percentages.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel2.colors)
    axs[1].set_title('Nose Conditions Percentage')

    # Throat conditions pie chart
    axs[2].pie(throat_percentages.values(), labels=throat_percentages.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
    axs[2].set_title('Throat Conditions Percentage')

    plt.tight_layout()
    st.pyplot(fig)



def plot_data_lable_train():
    # Data from the provided image
    categories = ['Tai (Ear)', 'Mũi (Nose)', 'Họng (Throat)']
    cases = [1071, 859, 855]
    total_cases = sum(cases)

    # Calculating the percentage of each category
    percentages = [x / total_cases * 100 for x in cases]

    # Creating the pie chart
    fig, ax = plt.subplots()
    ax.pie(percentages, labels=categories, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
    ax.set_title('Percentage of ENT Train')
    st.pyplot(fig)

def plot_data_lable_train_1():
    # Data from the provided categories and details
    ear_conditions = {'Viêm tai giữa ứ dịch': 180, 'Viêm tai giữa mãn tính': 180, 'Nấm ống tai ngoài': 182,
                    'Viêm ống tai ngoài cấp': 179, 'Viêm tai giữa cấp': 185, 'Tai bình thường':173 }
    nose_conditions = {'Viêm mũi dị ứng': 184, 'Viêm mũi xoang cấp': 172, 'Polyp mũi': 166,
                    'Vẹo vách ngăn': 165, 'Mũi bình thường': 162}
    throat_conditions = {'U vòm lành tính caVA': 176, 'Viêm thanh quản': 184, 'U lành tính': 174,
                        'Thanh quản bình thường': 178, 'Vòm bình thường': 173}

    # Calculating percentages
    total_ear = sum(ear_conditions.values())
    total_nose = sum(nose_conditions.values())
    total_throat = sum(throat_conditions.values())

    ear_percentages = {k: v / total_ear * 100 for k, v in ear_conditions.items()}
    nose_percentages = {k: v / total_nose * 100 for k, v in nose_conditions.items()}
    throat_percentages = {k: v / total_throat * 100 for k, v in throat_conditions.items()}

    # Creating pie charts
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Ear conditions pie chart
    axs[0].pie(ear_percentages.values(), labels=ear_percentages.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
    axs[0].set_title('Ear Conditions Percentage')

    # Nose conditions pie chart
    axs[1].pie(nose_percentages.values(), labels=nose_percentages.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel2.colors)
    axs[1].set_title('Nose Conditions Percentage')

    # Throat conditions pie chart
    axs[2].pie(throat_percentages.values(), labels=throat_percentages.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
    axs[2].set_title('Throat Conditions Percentage')

    plt.tight_layout()
    st.pyplot(fig)

def plot_data_train_data_lable():
    # Dữ liệu của bạn
    data_lable = 4314
    data_train = 2815

    # Tính phần trăm
    clean_percentage = data_train / data_lable * 100
    other_percentage = (data_lable - data_train) / data_lable * 100

    # Dữ liệu cho biểu đồ tròn
    labels = ['Data train', 'Data_lable']
    sizes = [clean_percentage, other_percentage]
    colors = ['lightcoral', 'lightskyblue']

    # Vẽ biểu đồ tròn
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    ax.set_title(' Data Clean vs Data_raw')
    st.pyplot(fig)




# Streamlit App
st.title("Image Dataset Analysis and Category Visualization")

# Dataset Path Input
dataset_path = st.text_input("Enter the dataset path:", "")

if dataset_path:
    try:
        image_info, total_size, class_distribution, unique_sizes = analyze_image_dataset(dataset_path)
        analysis_results = gather_analysis_results(image_info, total_size, class_distribution)
        st.subheader("Analysis Results")

        st.subheader("Unique Image Sizes")
        unique_sizes_list = [f"Kích thước {i+1}:\nWidth: {size[0]}, Height: {size[1]}" for i, size in enumerate(unique_sizes.keys())]
        st.write(unique_sizes_list)
        
        st.write(analysis_results)

        plot_analysis_results(analysis_results, unique_sizes)

        most_common_size, count = find_most_common_size(unique_sizes)
        st.write(f"The most common image size is {most_common_size[0]}x{most_common_size[1]} with {count} images.")

        # num_samples = st.slider("Number of sample images to display per class:", 1, 10, 5)
        # show_sample_images(dataset_path, num_samples)

    except FileNotFoundError:
        st.error(f"Dataset path '{dataset_path}' does not exist. Please check the path and try again.")

# Plot bar and pie charts for category data
plot_bar_and_pie_chart(),
plot_data_clean_pie_chart(),
plot_data_lable_pie_chart(),
plot_data_lable(),
plot_data_lable_1(),
plot_data_train_data_lable(),
plot_data_lable_train(),
plot_data_lable_train_1()


