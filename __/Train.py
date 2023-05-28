import os
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from Model.Model import UNet
from Dataset.AlphaPoseDataset import AlphaPoseDataSet
from __.Diffusion import GaussianDiffusion
from __.Scheduler import GradualWarmupScheduler


def do_train(device, config):
    # data
    train_dataset = AlphaPoseDataSet(config.path.train_dataset, pair_nums=config.data.train_pairs, image_size=config.data.size)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # model
    model = UNet(T=config.diffusion.T, in_ch=config.data.ch, model_ch=config.model.ch, out_ch=config.data.ch, ch_mult=config.model.ch_mult,
                 n_res_blocks=config.model.n_res_blocks, attn=config.model.attn, attn_heads=config.model.attn_heads).to(device)
    if config.model.compile:
        model = torch.compile(model)
    if config.train.pretrained:
        ckpt = torch.load(os.path.join(config.path.weight_dir, config.path.weight))
        model.load_state_dict(ckpt)
        print("load weight succeed:", config.path.weight)
    model.train()

    # optim
    optimizer = optim.Adam(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epoch, eta_min=0.)
    warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=config.optim.multiplier, 
                                              total_epoch=config.train.epoch // 10, after_scheduler=cosine_scheduler)
    
    # training
    diffusion = GaussianDiffusion(model, T=config.diffusion.T, beta_1=config.diffusion.beta_1, beta_T=config.diffusion.beta_T).to(device)
    print("training will start: \n  model: {}, pretrained: {}\n  epoch: [{}, {}]\n  batch_size: {}".
          format(model._get_name(), config.train.pretrained, config.train.start_epoch, config.train.epoch, config.train.batch_size))

    step = 0
    for epoch in range(config.train.start_epoch, config.train.end_epoch):
        with tqdm(train_dataloader) as tqdm_dataloader:
            for source_image, source_skeleton, target_image, target_skeleton in tqdm_dataloader:
                source_image, source_skeleton, target_image, target_skeleton = \
                    source_image.float().to(device), source_skeleton.float().to(device), \
                    target_image.float().to(device), target_skeleton.float().to(device)
                
                optimizer.zero_grad()
                loss = diffusion.train_model(target_image, [source_image, target_skeleton])
                loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                optimizer.step()

                step += 1
                tqdm_dataloader.set_postfix(ordered_dict={
                     "epoch": epoch, "loss": loss.item(), "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })

                if step % config.train.batch_size == 0:
                    x_T = torch.randn_like(target_image)
                    x_0 = diffusion.sample(x_T, [source_image, target_skeleton])
                    save_image(x_0, os.path.join(config.path.sample_dir, "Train", "train_sample_" + str(epoch) + "_" + str(step) + ".png"))

        warmup_scheduler.step()
        torch.save(model.state_dict(), os.path.join(config.path.weight_dir, "ckpt_" + str(epoch) + ".pt"))
        print(time.localtime(), "save model: ckpt_" + str(epoch) + ".pt")

    torch.save(model.state_dict(), os.path.join(config.path.weight_dir, "ckpt_last.pt"))
    print(time.localtime(), "train completed.")

@torch.no_grad()
def do_eval(device, config):
    # data
    eval_dataset = AlphaPoseDataSet(config.path.eval_dataset, pair_nums=config.data.eval_pairs, image_size=config.data.size)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.eval.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # model
    model = UNet(T=config.diffusion.T, in_ch=config.data.ch, model_ch=config.model.ch, out_ch=config.data.ch, ch_mult=config.model.ch_mult,
                 n_res_blocks=config.model.n_res_blocks, attn=config.model.attn, attn_heads=config.model.attn_heads).to(device)
    if config.model.compile:
        model = torch.compile(model)
    ckpt = torch.load(os.path.join(config.path.weight_dir, config.path.weight))
    model.load_state_dict(ckpt)
    model.eval()
    print("load weight succeed:", config.path.weight)

    # sample 
    diffusion = GaussianDiffusion(model, T=config.diffusion.T, beta_1=config.diffusion.beta_1, beta_T=config.diffusion.beta_T,
                                  w_s=config.diffusion.w_s, w_p=config.diffusion.w_p).to(device)
    print("sample will start: \n  model: {}, pretrained: {}\n  batch_size: {}".
          format(model._get_name(), config.path.weight, config.eval.batch_size))
    
    with tqdm(eval_dataloader) as tqdm_dataloader:
        for idx, (source_image, _, target_image, target_skeleton) in enumerate(tqdm_dataloader):
            source_image, target_image, target_skeleton = \
                source_image.float().to(device).float().to(device), target_image.float().to(device), target_skeleton.float().to(device)

            x_T = torch.randn_like(target_image)
            x_0 = diffusion.sample(x_T, [source_image, target_skeleton])

            sample_result = torch.cat([source_image, target_skeleton, target_image, x_0], dim=1)
            save_image(sample_result, os.path.join(config.path.sample_dir, "Eval", "train_sample_" + str(idx) + ".png"))

    print(time.localtime(), "eval completed.")
