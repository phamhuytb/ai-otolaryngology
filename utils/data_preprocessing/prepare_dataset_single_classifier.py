from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image


class ResizeMin:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img):
        width, height = img.size
        if width < height:
            new_width = self.min_size
            new_height = int(height * (self.min_size / width))
        else:
            new_height = self.min_size
            new_width = int(width * (self.min_size / height))
        return img.resize((new_width, new_height), Image.BILINEAR)
class ResizeAndCropTransform:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, img):
        # Đảm bảo kích thước nhỏ nhất là target_size
        width, height = img.size
        if width < self.target_size or height < self.target_size:
            scale = max(self.target_size / width, self.target_size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Cắt từ trung tâm
        left = (img.width - self.target_size) / 2
        top = (img.height - self.target_size) / 2
        right = (img.width + self.target_size) / 2
        bottom = (img.height + self.target_size) / 2
        img = img.crop((left, top, right, bottom))
        return img
def load_datasets(config):
    train_transform = transforms.Compose([


        ResizeMin(config['training']['img_size']+2),

        transforms.CenterCrop(config['training']['img_size']),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.ColorJitter(brightness=(0.85, 1.3)),
        transforms.ColorJitter(brightness=(0.95, 1.1), contrast=(0.95, 1.05)),
        transforms.RandomAffine(degrees=(-10, 10), shear=(-10, 10)),


        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([

        ResizeMin(config['training']['img_size']+2),
        transforms.CenterCrop(config['training']['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset = datasets.ImageFolder(root=config['training']['train_dataset_folder'], transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config['training']['val_dataset_folder'], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size_train'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size_val'], shuffle=False)

    return train_loader, val_loader, train_dataset.classes
