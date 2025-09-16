import os
from PIL import Image
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_image_dataset(dataset_path):
    image_info = defaultdict(list)
    total_size = 0
    class_distribution = defaultdict(int)

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(label_path, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        aspect_ratio = width / height
                        format = img.format
                        channels = 3 if img.mode == 'RGB' else (1 if img.mode == 'L' else len(img.getbands()))
                        img_array = np.array(img)
                        contrast = img_array.std()
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        class_distribution[label] += 1

                        image_info['sizes'].append((width, height))
                        image_info['aspect_ratios'].append(aspect_ratio)
                        image_info['formats'].append(format)
                        image_info['channels'].append(channels)
                        image_info['contrasts'].append(contrast)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return image_info, total_size, class_distribution

def gather_analysis_results(image_info, total_size, class_distribution):
    results = {
        "size_min": min(image_info['sizes']),
        "size_max": max(image_info['sizes']),
        "aspect_ratio_min": min(image_info['aspect_ratios']),
        "aspect_ratio_max": max(image_info['aspect_ratios']),
        "aspect_ratio_avg": np.mean(image_info['aspect_ratios']),
        "formats": dict(zip(*np.unique(image_info['formats'], return_counts=True))),
        "channels": dict(zip(*np.unique(image_info['channels'], return_counts=True))),
        "contrast_min": min(image_info['contrasts']),
        "contrast_max": max(image_info['contrasts']),
        "contrast_avg": np.mean(image_info['contrasts']),
        "class_distribution": dict(class_distribution),
        "total_size_mb": total_size / (1024 * 1024),
        "num_classes": len(class_distribution),
        "min_class": min(class_distribution, key=class_distribution.get),
        "max_class": max(class_distribution, key=class_distribution.get),
        "min_class_count": min(class_distribution.values()),
        "max_class_count": max(class_distribution.values())
    }
    return results

def plot_analysis_results(results):
    # Plotting Class Distribution
    class_distribution = results['class_distribution']
    plt.figure(figsize=(10, 5))
    plt.bar(class_distribution.keys(), class_distribution.values())
    plt.title('Class Distribution')
    plt.xticks(rotation=90)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.show()

    # Plotting Image Sizes
    sizes = results['size_min'], results['size_max']
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.bar(['Min Size', 'Max Size'], [sizes[0][0] * sizes[0][1], sizes[1][0] * sizes[1][1]])
    plt.title('Image Sizes')

    # Plotting Aspect Ratios
    aspect_ratios = results['aspect_ratio_min'], results['aspect_ratio_max'], results['aspect_ratio_avg']
    plt.subplot(2, 2, 2)
    plt.bar(['Min AR', 'Max AR', 'Avg AR'], aspect_ratios)
    plt.title('Aspect Ratios')

    # Plotting File Formats
    formats = results['formats']
    plt.subplot(2, 2, 3)
    plt.bar(formats.keys(), formats.values())
    plt.title('File Formats')

    # Plotting Number of Channels
    channels = results['channels']
    plt.subplot(2, 2, 4)
    plt.bar(channels.keys(), channels.values())
    plt.title('Number of Channels')

    plt.tight_layout()
    plt.show()

    # Plotting Image Contrast
    contrasts = results['contrast_min'], results['contrast_max'], results['contrast_avg']
    plt.figure(figsize=(10, 5))
    plt.bar(['Min Contrast', 'Max Contrast', 'Avg Contrast'], contrasts)
    plt.title('Image Contrast')
    plt.show()

def show_sample_images(dataset_path, num_samples=5):
    sample_images = []
    class_labels = os.listdir(dataset_path)
    for label in class_labels:
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue

        files = [os.path.join(label_path, f) for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        sample_images.extend([(label, f) for f in files[:num_samples]])

    plt.figure(figsize=(15, 15))
    for i, (label, file_path) in enumerate(sample_images[:num_samples * len(class_labels)]):
        img = Image.open(file_path)
        plt.subplot(len(class_labels), num_samples, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis('off')
    plt.show()

# Thay đổi đường dẫn tới tập dữ liệu của bạn
dataset_path = "path/to/your/dataset"
image_info, total_size, class_distribution = analyze_image_dataset(dataset_path)
analysis_results = gather_analysis_results(image_info, total_size, class_distribution)

# Vẽ các biểu đồ từ kết quả phân tích
plot_analysis_results(analysis_results)

# Hiển thị một số ảnh mẫu
show_sample_images(dataset_path)
