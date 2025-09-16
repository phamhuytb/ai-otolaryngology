from PIL import Image

# resize image to min size, exp min size = 250, 1000x500 = 500x250
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