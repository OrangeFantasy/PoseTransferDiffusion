import os
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser

from diffusion.utils import instantiate_from_config


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/diffusion_config.yaml")
     
    return parser


def prepare_data(data_config, vae_model, device, save_dir):
    dataset = instantiate_from_config(data_config)

    out_dirs = ["encoding_img", "encoding_pose"]
    for idx in tqdm(range(dataset.__len__()), total=dataset.__len__(), colour="#808f57", desc=save_dir):
        paths = dataset.data[idx]

        img_save_path = paths["source_image"].replace("img", os.path.join(save_dir, out_dirs[0]))
        if not os.path.exists(os.path.dirname(img_save_path)):
            os.makedirs(os.path.dirname(img_save_path))
        
        pose_save_path = paths["source_skeleton"].replace("keypoints", os.path.join(save_dir, out_dirs[1]))
        if not os.path.exists(os.path.dirname(pose_save_path)):
            os.makedirs(os.path.dirname(pose_save_path))
        
        if os.path.exists(img_save_path + ".npy"):
            continue

        src_img, src_pose, _, _ = dataset.__getitem__(idx)
        src_img, src_pose = src_img.unsqueeze(0).to(device), src_pose.unsqueeze(0).to(device)

        img_enc = vae_model.encode(src_img).sample().squeeze().detach().cpu().numpy()
        pose_enc = vae_model.encode(src_pose).sample().squeeze().detach().cpu().numpy()

        np.save(img_save_path, img_enc)
        np.save(pose_save_path, pose_enc)


if __name__ == "__main__":
    print("[INFO] Prepare data encoding by Autoencoder...")
    
    parser = get_parser()
    opt = parser.parse_args()
    config = OmegaConf.load(opt.config)
    
    import torch
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    vae_config = config.model.params.first_stage_config
    vae_model = instantiate_from_config(vae_config).to(device).eval()
    
    print("[INFO] Prepare train data: ")
    data_config = config.raw_data_config
    # prepare_data(data_config, vae_model, device, "train")
    
    print("[INFO] Prepare test data: ")
    data_config.params.is_train = False
    prepare_data(data_config, vae_model, device, "test")
