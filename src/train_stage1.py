import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

import torchvision.datasets as datasets

import cv2 as cv2
import numpy as np
# import pandas as pd
from PIL import Image
import os
import utils
from datetime import datetime
import random
import transporter
from transporter import Block

from datetime import datetime
import torch.utils.data
from torchvision import transforms
import PIL

task_info = {
    "name": "cartpole, balance",

    "replay_buffer_filename": "replay_buffer.npy",

    # Episode length has to do with frame skip: 1000 / 4 = 250
    "episode_length": 250,

    # This has to do with how many iters do you run for sampling.
    "num_trajectories": 20,

    "k": 12

    # For environments with moving goals in different environments,
    # Please set ignore_time_limit to true to blend with different trajectories

    # "ignore_time_limit": True
}

task_name = task_info["name"]
replay_buffer_filename = task_info["replay_buffer_filename"]
episode_length = task_info["episode_length"]
num_trajectories = task_info["num_trajectories"]
ignore_time_limit = task_info.get("ignore_time_limit", False)
keypoint_std = task_info.get("keypoint_std", 0.1)

# Most training environments we use do not have much noise and it is relatively easy to extract foreground, 
# so a larger k may benefit more (having some background does not hurt because if we have 
# all background, we go back to SAC baseline, but missing information hurts the learning efficiency).
k = task_info.get("k", 24)
print("Task:", task_name)
print("Replay buffer:", replay_buffer_filename)

replay_buffer = np.load(replay_buffer_filename, allow_pickle=True)
if len(replay_buffer.shape) == 4:
    obses = np.transpose(replay_buffer, (0, 3, 1, 2))
    next_obses = None
else:
    images = replay_buffer.item()
    obses, next_obses = images["obses"], images["next_obses"]

assert obses.shape[0] == episode_length * num_trajectories

config = utils.ConfigDict({})
config.dataset_obses = obses
config.batch_size = 64
config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config.image_channels = 3
config.k = k
config.num_iterations = int(3000)
config.learning_rate = 1e-3
config.learning_rate_decay_rate = 0.95
config.learning_rate_decay_every_n_steps = int(200)
config.report_every_n_steps = 200

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
config.log_file = os.path.join('runs', current_time + '_dm_control.pth')
print("Save to:", config.log_file)

def convert_n_t_to_idx(n, t):
    return n * episode_length + t

class Dataset(object):
    def __init__(self, obses, next_obses, transform=None):
        self.obses = obses
        self.next_obses = next_obses
        self._transform = transform

    def __len__(self):
        raise NotImplementedError

    def get_image(self, n, t):
        return self.obses[convert_n_t_to_idx(n, t)]

    @property
    def num_trajectories(self):
        return num_trajectories
  
    @property
    def num_timesteps(self):
        return episode_length
    
    def __getitem__(self, idx):
        n, n2, t, tp1 = idx
        # Note: de-noise trick can be implemented here to reduce the influence of video noise
        imt = self.get_image(n, t)
        imtp1 = self.get_image(n2, tp1)
        if self._transform is not None:
            imt, imtp1, imtp1_orig = self._transform(imt, imtp1)

        return imt, imtp1, imtp1_orig

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self._dataset = dataset

    def __iter__(self):
        while True:
            if ignore_time_limit and random.random() > 0.5:
                n = np.random.randint(self._dataset.num_trajectories)
                n2 = np.random.randint(self._dataset.num_trajectories)
                num_images = self._dataset.num_timesteps
                t_ind = np.random.randint(0, num_images)
                tp1_ind = np.random.randint(0, num_images)
                yield n, n2, t_ind, tp1_ind
            else:
                n = np.random.randint(self._dataset.num_trajectories)
                num_images = self._dataset.num_timesteps
                t_ind = np.random.randint(0, num_images - 20)
                tp1_ind = t_ind + np.random.randint(20)
                yield n, n, t_ind, tp1_ind

    def __len__(self):
        raise NotImplementedError

def _get_model(config):
    feature_encoder = transporter.FeatureEncoder(config.image_channels)
    pose_regressor = transporter.PoseRegressor(config.image_channels, config.k)
    refine_net = transporter.RefineNet(config.image_channels)

    return transporter.Transporter(feature_encoder, pose_regressor, refine_net, std=keypoint_std)

def apply_all(imgs, func, *args, except_last=False):
    if except_last:
        return [func(img, *args) for img in imgs[:-1]] + [imgs[-1]]
    return [func(img, *args) for img in imgs]

def _transform(img1, img2):
    #img1, img2 = torch.tensor(img1)/255., torch.tensor(img2)/255.
    
    imgs = [img1, img2, img2]
    imgs = apply_all(imgs, np.transpose, (1,2,0))
    imgs = apply_all(imgs, TF.to_pil_image)

    # random crop or affine may be a good choice
    
    imgs = apply_all(imgs, TF.to_tensor)
    imgs = apply_all(imgs, TF.convert_image_dtype)
    return imgs

def _get_data_loader(config):
    dataset = Dataset(obses, next_obses, transform=_transform)
    sampler = Sampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, sampler=sampler, pin_memory=True, num_workers=4)
    return loader

def get_encoder(in_channels, return_net_channels=False):
    net = nn.Sequential(
            Block(in_channels, 32, kernel_size=(7, 7), stride=1, padding=3), # 1
            Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            Block(32, 64, kernel_size=(3, 3), stride=2),  # 3
            Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            Block(64, 128, kernel_size=(3, 3), stride=2), # 5
            Block(128, 128, kernel_size=(3, 3), stride=1),  # 6        
        )
    
    if return_net_channels:
        return net, 128
    return net

def get_decoder(num_channels):
    return nn.Sequential(
        Block(128, 128, kernel_size=(3, 3), stride=1), # 6 
        Block(128, 64, kernel_size=(3, 3), stride=1), # 5
        nn.UpsamplingBilinear2d(scale_factor=2),
        Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
        Block(64, 32, kernel_size=(3, 3), stride=1),  # 3
        nn.UpsamplingBilinear2d(scale_factor=2),
        Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
        Block(32, num_channels, kernel_size=(7, 7), stride=1, padding=3), # 1
    )
transporter.get_encoder = get_encoder
transporter.get_decoder = get_decoder

model = _get_model(config)
model = model.to(config.device)
loader = _get_data_loader(config)

optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    config.learning_rate_decay_every_n_steps,
    gamma=config.learning_rate_decay_rate)

model.train()
for itr, (xt, xtp1, xtp1_original) in enumerate(loader):
    xt = xt.to(config.device).float()
    xtp1 = xtp1.to(config.device).float()
    xtp1_original = xtp1_original.to(config.device).float()
    optimizer.zero_grad()
    reconstruction = model(xt, xtp1)

    loss = torch.nn.functional.mse_loss(reconstruction, xtp1_original)
    loss.backward()

    optimizer.step()
    scheduler.step()
    if itr % config.report_every_n_steps == 0:
        print("Save model at", itr)
        torch.save(model.state_dict(), os.path.join(config.log_file))
    
    if itr > config.num_iterations:
        break

