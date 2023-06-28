import os
import cv2
import math
import numpy as np
import torch
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as vf


class DeepFashionDataset(Dataset):
    def __init__(self, root, image_size, is_train: bool = True):
        super().__init__()

        self.root = root
        self.image_size = image_size
        self.raw_size = [256, 256]
        self.data = self.get_paths_from("train_pairs.txt" if is_train else "test_pairs.txt")

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Draw skeleton setting.
        self.limb_seq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12],
                         [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
                                [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
                                [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        self.line_width = int(self.image_size[0] / self.raw_size[0] * 3) + 1
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path = self.data[index]
        
        source_image = self.load_image(path["source_image"])
        source_skeleton = self.load_skeleton_from_keypoints(path["source_skeleton"])
        target_image = self.load_image(path["target_image"])
        target_skeleton = self.load_skeleton_from_keypoints(path["target_skeleton"])

        return source_image, source_skeleton, target_image, target_skeleton

    def get_paths_from(self, pairs_file):
        file = open(os.path.join(self.root, pairs_file))
        lines = file.readlines()
        file.close()

        image_paths = []
        for item in lines:
            dict_item = {}
            item = item.strip().split(',')
            dict_item["source_image"] = os.path.join(self.root, item[0])
            dict_item["source_skeleton"] = os.path.join(self.root, self.replace_image_to_keypoints(item[0]))
            dict_item["target_image"] = os.path.join(self.root, item[1])
            dict_item["target_skeleton"] = os.path.join(self.root, self.replace_image_to_keypoints(item[1]))
            image_paths.append(dict_item)
        return image_paths
    
    def load_image(self, path):
        image = Image.open(path)
        tensor = self.image_transform(image)
        return tensor
    
    def load_skeleton_from_keypoints(self, path):
        w, h = self.image_size
        canvas = np.zeros([h, w, 3], dtype=np.uint8)
        keypoints = np.loadtxt(path)
        keypoints = self.trans_keypoints(keypoints)

        for idx in range(18):
            x, y = keypoints[idx]
            if x == -1 or y == -1:
                continue
            cv2.circle(canvas, center=[int(x), int(y)], radius=self.line_width, color=self.colors[idx], thickness=-1)

        # joint_list = []
        for idx in range(17):
            x1, y1 = keypoints[self.limb_seq[idx][0] - 1]
            x2, y2 = keypoints[self.limb_seq[idx][1] - 1]
            if -1 in [x1, y1, x2, y2]:
                # joint_list.append(np.zeros([h, w, 1], dtype=np.uint8))
                continue
            
            curr_canvas = canvas.copy()
            mean_x, mean_y = np.mean([x1, x2]), np.mean([y1, y2])
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            angle = math.degrees(math.atan2(y1 - y2, x1 - x2))
            polygon = cv2.ellipse2Poly((int(mean_x), int(mean_y)), (int(length / 2), self.line_width), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(curr_canvas, polygon, self.colors[idx])
            canvas = cv2.addWeighted(canvas, 0.5, curr_canvas, 0.5, 0)

            # joint = np.zeros([h, w, 1], dtype=np.uint8)
            # cv2.fillConvexPoly(joint, polygon, color=255)
            # joint_list.append(joint)
        pose_tensor = vf.to_tensor(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
        return pose_tensor

        # dist_tensor = []
        # for joint in joint_list:
        #     dist = cv2.distanceTransform(255 - joint, distanceType=cv2.DIST_L1, maskSize=3)
        #     dist = (dist - np.min(dist)) / (np.max(dist) - np.min(dist) + 1e-10)

        #     dist_tensor.append(vf.to_tensor(Image.fromarray(dist)))
        # dist_tensor = torch.cat(dist_tensor, dim=0)

        # skeleton_tensor = torch.cat([pose_tensor, dist_tensor], dim=0)
        # return skeleton_tensor

    def trans_keypoints(self, keypoints):
        missing_keypoint_index = keypoints == -1    

        _w = 1. / self.raw_size[0] * self.image_size[0]
        _h = 1. / self.raw_size[1] * self.image_size[1]

        keypoints[:, 0] *= _w
        keypoints[:, 1] *= _h
        keypoints[missing_keypoint_index] = -1
        return keypoints

    @staticmethod
    def replace_image_to_keypoints(path: str) -> str:
        return path.replace("img", "keypoints").replace(".jpg", ".txt")


# class DeepFashionDataset_FromEncoding(Dataset):
#     def __init__(self, root, pairs_num: int = -1, is_train: bool = True):
#         self.root = root
#         self.data_paths = self.get_paths_from("train_pairs.txt" if is_train else "test_pairs.txt")
        
#         if pairs_num != -1:
#             self.data_paths = random.sample(self.data_paths, k=pairs_num)

#         self.size = len(self.data_paths)
    
#     def __len__(self):
#         return self.size
    
#     def __getitem__(self, index):
#         paths = self.data_paths[index]
#         src_img = torch.from_numpy(np.load(paths[0]))
#         src_pose = torch.from_numpy(np.load(paths[1]))
#         tgt_img = torch.from_numpy(np.load(paths[2]))
#         tgt_pose = torch.from_numpy(np.load(paths[3]))

#         return src_img, src_pose, tgt_img, tgt_pose

#     def get_paths_from(self, pairs_file):
#         file = open(os.path.join(self.root, pairs_file))
#         lines = file.readlines()
#         file.close()

#         image_paths = []
#         for item in lines:
#             paths = []
#             item = item.strip().split(",")

#             paths.append(os.path.join(self.root, self._to_img_encoding_path(item[0])))
#             paths.append(os.path.join(self.root, self._to_pose_encoding_path(item[0])))
#             paths.append(os.path.join(self.root, self._to_img_encoding_path(item[1])))
#             paths.append(os.path.join(self.root, self._to_pose_encoding_path(item[1])))
#             image_paths.append(paths)
#         return image_paths
    
#     @staticmethod
#     def _to_img_encoding_path(path: str) -> str:
#         return path.replace("img", "encoding_img") + ".npy"
    
#     @staticmethod
#     def _to_pose_encoding_path(path: str) -> str:
#         return path.replace("img", "encoding_pose").replace(".jpg", ".txt") + ".npy"


# class DeepFashionDataset_FromMemory(Dataset):
#     def __init__(self, root, pairs_num: int = -1, is_train: bool = True):
#         self.root = root
#         self.data = self.get_all_data_from("train_pairs.txt" if is_train else "test_pairs.txt")
        
#         if pairs_num != -1:
#             self.data = random.sample(self.data, k=pairs_num)

#         self.size = len(self.data)
    
#     def __len__(self):
#         return self.size
    
#     def __getitem__(self, index):
#         data = self.data[index]
#         # src_img = torch.from_numpy(np.load(paths[0]))
#         # src_pose = torch.from_numpy(np.load(paths[1]))
#         # tgt_img = torch.from_numpy(np.load(paths[2]))
#         # tgt_pose = torch.from_numpy(np.load(paths[3]))

#         return self.data[index]

#     def get_all_data_from(self, pairs_file):
#         file = open(os.path.join(self.root, pairs_file))
#         lines = file.readlines()
#         file.close()

#         image_data = []
#         for item in lines:
#             data = []
#             item = item.strip().split(",")

#             data.append(torch.from_numpy(np.load(os.path.join(self.root, self._to_img_encoding_path(item[0])))))
#             data.append(torch.from_numpy(np.load(os.path.join(self.root, self._to_pose_encoding_path(item[0])))))
#             data.append(torch.from_numpy(np.load(os.path.join(self.root, self._to_img_encoding_path(item[1])))))
#             data.append(torch.from_numpy(np.load(os.path.join(self.root, self._to_pose_encoding_path(item[1])))))
#             image_data.append(data)
#         return image_data
    
#     @staticmethod
#     def _to_img_encoding_path(path: str) -> str:
#         return path.replace("img", "encoding_img") + ".npy"
    
#     @staticmethod
#     def _to_pose_encoding_path(path: str) -> str:
#         return path.replace("img", "encoding_pose").replace(".jpg", ".txt") + ".npy"


# if __name__ == "__main__":
#     root = r"E:/_Project/_Dataset/In-shop Clothes Retrieval Benchmark"
#     # train_dataset = DeepFashionDataset_1(root, image_size=[64, 64])

#     # paths = train_dataset.data[512]
#     # test_pose_path = paths["target_skeleton"]  #.replace("img", "keypoints").replace(".jpg", ".txt")
#     # train_dataset.load_skeleton_from_keypoints(test_pose_path)

#     # dataset = DeepFashionDataset_2(root, "train_img_pairs.csv", "train_pose_maps", pairs_nums=100)
#     # a, b, c, d = dataset.__getitem__(0)

#     # import torch
#     # from torchvision.utils import save_image
#     # save_image(torch.cat([a, b, c, d], dim=-1), "./png")

#     from torchvision.utils import save_image
#     dataset = DeepFashionDataset_FromEncoding(root, pairs_num=200)
#     src_img, src_pose, tgt_img, tgt_pose = dataset.__getitem__(100)
#     save_image(torch.cat([src_img, src_pose, tgt_img, tgt_pose], dim=-1), "./1.png")

#     print()