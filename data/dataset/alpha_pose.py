import os
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AlphaPoseDataSet(Dataset):
    def __init__(self, root: str, pair_nums: int, image_size: int) -> None:
        super().__init__()

        self.root = root
        self.pair_nums = pair_nums
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.data_pair = self.get_data_pair()

    def __len__(self) -> int:
        return len(self.data_pair)
    
    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        paths = self.data_pair[index]
        source_image, source_skeleton = self.load_and_transform_image(paths[0])
        target_image, target_skeleton = self.load_and_transform_image(paths[1])
        
        return source_image, source_skeleton, target_image, target_skeleton
    
    def get_data_pair(self) -> list:
        data_paths = []
        data_classes, data_nums = 0, 0
        for dir in os.listdir(self.root):
            data_classes += 1
            dir_path = os.path.join(self.root, dir)

            curr_dir_img_paths = []
            for image in os.listdir(dir_path):
                data_nums += 1
                image_path = os.path.join(dir_path, image)
                curr_dir_img_paths.append(image_path)
            
            data_paths.append(curr_dir_img_paths)

        probabilities = []
        for dir in data_paths:
            probabilities.append(len(dir) / data_nums)
        
        data_pair = []
        for _ in range(self.pair_nums):
            random_class = np.random.choice(data_classes, size=1, p=probabilities)[0]
            random_pair_index = np.random.choice(len(data_paths[random_class]), size=2)
            data_pair.append([data_paths[random_class][random_pair_index[0]], data_paths[random_class][random_pair_index[1]]])
        
        return data_pair
    
    def load_and_transform_image(self, path: str) -> tuple[Tensor, Tensor]:
        image_and_skeleton = Image.open(path).convert('RGB')
        w, h = image_and_skeleton.size

        image = image_and_skeleton.crop([0, 0, w // 2, h])
        image = self.transform(image)
        skeleton = image_and_skeleton.crop([w // 2, 0, w, h])
        skeleton = self.transform(skeleton)

        return image, skeleton
    

if __name__ == "__main__":
    dataset = AlphaPoseDataSet("D:\\_Project\\Dataset\\AlphaPoseData\\train\\data", 10)
    source_image, source_skeleton, target_image, target_skeleton = dataset.__getitem__(5)
    print(source_image.shape, source_skeleton.shape, target_image.shape, target_skeleton.shape)