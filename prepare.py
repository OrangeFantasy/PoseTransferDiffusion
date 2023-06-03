import os
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from diffusion.utils import instantiate_from_config


def prepare_data():
    config = OmegaConf.load("configs/diffusion_config.yaml")

    vae_config = config.model.params.first_stage_config
    vae_model = instantiate_from_config(vae_config).cuda().eval()

    data_config = config.raw_data_config
    dataset = instantiate_from_config(data_config)

    out_dirs = ["encoding_img", "encoding_pose"]
    iterator = tqdm(range(dataset.__len__()), colour="#00ff00")
    for idx in iterator:
        paths = dataset.data[idx]

        img_path = paths["source_image"].replace("img", out_dirs[0])
        if not os.path.exists(os.path.dirname(img_path)):
            os.makedirs(os.path.dirname(img_path))
        
        pose_path = paths["source_skeleton"].replace("keypoints", out_dirs[1])
        if not os.path.exists(os.path.dirname(pose_path)):
            os.makedirs(os.path.dirname(pose_path))
        
        if os.path.exists(img_path + ".npy"):
            continue

        src_img, src_pose, _, _ = dataset.__getitem__(idx)
        src_img, src_pose = src_img.unsqueeze(0).cuda(), src_pose.unsqueeze(0).cuda()

        img_enc = vae_model.encode(src_img).sample().squeeze().detach().cpu().numpy()
        pose_enc = vae_model.encode(src_pose).sample().squeeze().detach().cpu().numpy()

        np.save(img_path, img_enc)
        np.save(pose_path, pose_enc)


if __name__ == "__main__":
    print("[INFO] Prepare data encoding by autoencoder: ")
    prepare_data()
