import os
from PIL import Image
import numpy as np
from collections import defaultdict
import yaml

# def load_config(config_path):
#     with open(config_path, 'r') as file:
#         return yaml.safe_load(file)

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

def print_analysis_results(image_info, total_size, class_distribution):
    print("Kích thước ảnh:")
    print(f"  Min: {min(image_info['sizes'])}")
    print(f"  Max: {max(image_info['sizes'])}")
    print(f"  Average: {tuple(np.mean(image_info['sizes'], axis=0).astype(int))}")

    print("\nTỷ lệ khung hình:")
    print(f"  Min: {min(image_info['aspect_ratios']):.2f}")
    print(f"  Max: {max(image_info['aspect_ratios']):.2f}")
    print(f"  Average: {np.mean(image_info['aspect_ratios']):.2f}")

    print("\nĐịnh dạng file:")
    for format, count in zip(*np.unique(image_info['formats'], return_counts=True)):
        print(f"  {format}: {count}")

    print("\nSố lượng kênh màu:")
    for channels, count in zip(*np.unique(image_info['channels'], return_counts=True)):
        print(f"  {channels}: {count}")

    print("\nChất lượng ảnh (độ tương phản):")
    print(f"  Min: {min(image_info['contrasts']):.2f}")
    print(f"  Max: {max(image_info['contrasts']):.2f}")
    print(f"  Average: {np.mean(image_info['contrasts']):.2f}")

    print("\nPhân phối dữ liệu:")
    for label, count in class_distribution.items():
        print(f"  {label}: {count}")

    print(f"\nTổng dung lượng: {total_size / (1024 * 1024):.2f} MB")

    print("\nTính đa dạng của dữ liệu:")
    print(f"  Số lượng lớp: {len(class_distribution)}")
    print(f"  Lớp có ít mẫu nhất: {min(class_distribution, key=class_distribution.get)} ({min(class_distribution.values())} mẫu)")
    print(f"  Lớp có nhiều mẫu nhất: {max(class_distribution, key=class_distribution.get)} ({max(class_distribution.values())} mẫu)")

if __name__ == "__main__":
    # Đường dẫn đến file config
      # Sử dụng đường dẫn tuyệt đối đến file config_train_file.yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(current_dir, "..", "config")

    def load_config(config_name):
        config_file_path = os.path.join(CONFIG_PATH, config_name)
        with open(config_file_path, 'r') as file:
            config = yaml.safe_load(file)
        return config


    

    config = load_config("config_data_process.yaml")
    
    dataset_path = config['paths']['dataset_path']
    
    image_info, total_size, class_distribution = analyze_image_dataset(dataset_path)
    print_analysis_results(image_info, total_size, class_distribution)