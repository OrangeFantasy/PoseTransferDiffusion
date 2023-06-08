import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class DeepFashionDataset_FromEncoding(Dataset):
    def __init__(self, root, pairs_num: int = -1, is_train: bool = True):
        self.root = root
        self.data_paths = self.get_paths_from("train_pairs.txt" if is_train else "test_pairs.txt")
        
        if pairs_num != -1:
            self.data_paths = random.sample(self.data_paths, k=pairs_num)

        self.size = len(self.data_paths)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        paths = self.data_paths[index]
        src_img = torch.from_numpy(np.load(paths[0]))
        src_pose = torch.from_numpy(np.load(paths[1]))
        tgt_img = torch.from_numpy(np.load(paths[2]))
        tgt_pose = torch.from_numpy(np.load(paths[3]))

        return src_img, src_pose, tgt_img, tgt_pose

    def get_paths_from(self, pairs_file):
        file = open(os.path.join(self.root, pairs_file))
        lines = file.readlines()
        file.close()

        image_paths = []
        for item in lines:
            paths = []
            item = item.strip().split(",")

            paths.append(os.path.join(self.root, self._to_img_encoding_path(item[0])))
            paths.append(os.path.join(self.root, self._to_pose_encoding_path(item[0])))
            paths.append(os.path.join(self.root, self._to_img_encoding_path(item[1])))
            paths.append(os.path.join(self.root, self._to_pose_encoding_path(item[1])))
            image_paths.append(paths)
        return image_paths
    
    @staticmethod
    def _to_img_encoding_path(path: str) -> str:
        return path.replace("img", "encoding_img") + ".npy"
    
    @staticmethod
    def _to_pose_encoding_path(path: str) -> str:
        return path.replace("img", "encoding_pose").replace(".jpg", ".txt") + ".npy"
