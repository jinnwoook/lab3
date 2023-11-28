from PIL import Image
from torch.utils.data import Dataset
from lib.utils import load_json


CIFAR10_MEAN = (0.491, 0.482, 0.447)
CIFAR10_STD = (0.247, 0.244, 0.262)


def read_image(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class Cifar10Labeled(Dataset):
    def __init__(self, root, path, transform=None, target_transform=None):
        self.root = root
        self.data = load_json(path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = f'{self.root}/{self.data[index]["image"]}'
        image = read_image(image_path)
        target = self.data[index]["label"]


        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target


class Cifar10Unlabeled(Dataset):
    def __init__(self, root, path, transform=None, target_transform=None):
        self.root = root
        self.data = load_json(path)  
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = f'{self.root}/{self.data[index]["image"]}'
        image = read_image(image_path)  

        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(image)
        
        return image, target


class Cifar10UnlabeledFast(Dataset):
    def __init__(self, image_path, transform=None, target_transform=None):
        self.images = np.load(image_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        image = Image.fromarray(self.images[index])
        
        image_copy = image.copy()

        
        if self.transform is not None:
            image = self.transform(image)

        
        if self.target_transform is not None:
            image_copy = self.target_transform(image_copy)

        
        return image, image_copy
