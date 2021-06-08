import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
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

task_info = {
    "name": "cartpole, balance",

    "replay_buffer_filename": "replay_buffer.npy",

    # This threshold is typically stable (around 0.35 ~ 0.5) and typically not affected
    # by different background color in different DMC environments
    # since we apply causal inference to remove background color
    "foreground_threshold": 0.35,

    # Please set it to your checkpoint:
    # "keypoint_checkpoint": "runs/Oct31_08-32-43_dm_control.pth",
    
    "keypoint_checkpoint": "runs/May31_00-49-36_dm_control.pth",

    # This needs to match the k used in stage 1.
    "k": 12,

    # This creates a simulated distractor and is useful for environments with mirroring floors.
    "distracor_probability": 0,

    # If you find the augmentation too weak (e.g. performs not well in videos), enable this
    "add_random_boxes": False,
    "boxes_kwargs": {
        "boxes": 5, 
        "mu": 5, 
        "sigma": 5, 
        "size_min": 1, 
        "size_max": 10
    },

    # This gives different weights to foreground and background to ask them to prefer which kind of misclassification; here we ask it to classify as foreground if it misclassifies because it will not lead to missing information.
    "pos_weight": 15,

    "episode_length": 250,
    "num_trajectories": 20,
    
    # This adds noise to background.
    "noise_level": 0.1,

    # You can use this to override configs
    "config_override": {
        # "num_iterations": 1000
    }
}

task_name = task_info["name"]

replay_buffer_filename = task_info["replay_buffer_filename"]
keypoint_checkpoint = task_info["keypoint_checkpoint"]
foreground_threshold = task_info["foreground_threshold"]
episode_length = task_info["episode_length"]
num_trajectories = task_info["num_trajectories"]
distracor_probability = task_info["distracor_probability"]
add_random_boxes = task_info["add_random_boxes"]
boxes_kwargs = task_info["boxes_kwargs"]
keypoint_std = task_info.get("keypoint_std", 0.25)
dilate = task_info.get("dilate", False)
erode = task_info.get("erode", False)
alpha_high = task_info.get("alpha_high", 1)
alpha_low = task_info.get("alpha_low", 0)
noise_level = task_info["noise_level"]
weak_foreground_augment = task_info.get("weak_foreground_augment", False)
abs_comparison = task_info.get("abs_comparison", False) # This is for compatibility.
k = task_info.get("k", 24)
neg_weight = task_info.get("neg_weight", 1.0)
pos_weight = task_info.get("pos_weight", 1.0)

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
checkpoint_path = os.path.join('runs', current_time + '_dm_control.pth')

print("Task:", task_name)
print("Replay buffer:", replay_buffer_filename)
print("Checkpoint:", checkpoint_path)

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
config.dataset_next_obses = next_obses
config.batch_size = 64

config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config.image_channels = 3
config.num_iterations = int(3000) # 5000 samples is about 78 steps
config.learning_rate = 1e-3
config.learning_rate_decay_rate = 0.95
config.learning_rate_decay_every_n_steps = int(200)
config.report_every_n_steps = 100

config.log_file = checkpoint_path

if "config_override" in task_info:
    for item, value in task_info["config_override"].items():
        setattr(config, item, value)

data_dir = "./places365_standard"
image_size = 84
num_workers = 16
external_dataset_batch_size = config.batch_size * 3 # since we have 3 frames
fp = os.path.join(data_dir, 'train')
places_dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(fp, transforms.Compose([
    transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])), batch_size=external_dataset_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

places_iter = None

def _get_places_batch():
    global places_iter
    try:
        imgs, _ = next(places_iter)
        if imgs.size(0) < batch_size:
            places_iter = iter(places_dataloader)
            imgs, _ = next(places_iter)
    except StopIteration:
        places_iter = iter(places_dataloader)
        imgs, _ = next(places_iter)
    return imgs.cuda(config.device)

def _reset_places():
    global places_iter
    places_iter = iter(places_dataloader)

class AdapterEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        
        in_channels = 16
        
        self.front = nn.Sequential(
            Block(obs_shape[0], in_channels, kernel_size=(3, 3), stride=2, padding=1), # 1
            Block(in_channels, 16, kernel_size=(3, 3), stride=1),  # 2
        )
        
        self.attention = Block(16, 16, kernel_size=(3, 3), stride=2) # 3
        self.middle = Block(16, 16, kernel_size=(3, 3), stride=2) # 3
        
        self.last = Block(16, 16, kernel_size=(3, 3), stride=1)  # 4
        
    def forward(self, x):
        after_front = self.front(x)
        
        after_attention = self.attention(after_front).sigmoid()
        after_middle = self.middle(after_front)
        
        return self.last(after_middle * after_attention) # attention will broadcast in channels

def adapter_get_encoder(obs_shape, num_filters):
    return AdapterEncoder(obs_shape)

def adapter_get_decoder(obs_shape, num_filters):
    return nn.Sequential(
        Block(16, 16, kernel_size=(3, 3), stride=1),  # 4
        Block(16, 16, kernel_size=(3, 3), stride=1),  # 3
        nn.UpsamplingBilinear2d(scale_factor=2),
        Block(16, 16, kernel_size=(3, 3), stride=1),  # 2
        Block(16, 1, kernel_size=(3, 3), stride=1, padding=1), # 1
        nn.UpsamplingBilinear2d(scale_factor=2)
    )

def _get_adapter_model(config):
    #obs_shape = (9, 84, 84)
    obs_shape = (3, 84, 84)
    num_filters = 16
    encoder = adapter_get_encoder(obs_shape, num_filters).to(device=config.device)
    decoder = adapter_get_decoder(obs_shape, num_filters).to(device=config.device)
    return encoder, decoder

batch_size = 32
image_channels = 3
num_features = 32

def transporter_get_encoder(in_channels, return_net_channels=False):
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

def transporter_get_decoder(num_channels):
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
transporter.get_encoder = transporter_get_encoder
transporter.get_decoder = transporter_get_decoder

feature_encoder = transporter.FeatureEncoder(image_channels)
pose_regressor = transporter.PoseRegressor(image_channels, k)
refine_net = transporter.RefineNet(image_channels)

model = transporter.Transporter(
    feature_encoder, pose_regressor, refine_net
)

model.load_state_dict(
    torch.load(keypoint_checkpoint, map_location='cpu')
)
model.to(device=config.device)
model.eval()

def mask(keypoints, features):
    out = features * keypoints.max(dim=1, keepdim=True)[0]
    return out

T = 3

def convert_n_t_to_idx(n, t):
    return n * episode_length + t

class Dataset(object):
    def __init__(self, obses, transform=None):
        self.obses = obses
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
        n, t = idx
        imgs = [self.get_image(n, t+ind) for ind in range(T)]
        if self._transform:
            imgs = apply_all(imgs, self._transform)
        return np.vstack(imgs)

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self._dataset = dataset

    def __iter__(self):
        while True:
            n = np.random.randint(self._dataset.num_trajectories)
            num_images = self._dataset.num_timesteps
            t_ind = np.random.randint(0, num_images - T)
            yield n, t_ind

    def __len__(self):
        raise NotImplementedError

def apply_all(imgs, func, *args, except_last=False, **kwargs):
    if except_last:
        return [func(img, *args, **kwargs) for img in imgs[:-1]] + [imgs[-1]]
    return [func(img, *args, **kwargs) for img in imgs]

def transform(img):
    img = img / 255.
    return img

def _get_data_loader(config):
    dataset = Dataset(obses, transform=transform)
    sampler = Sampler(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, sampler=sampler, pin_memory=True, num_workers=4)
    return loader

def random_crop_cuda(x, size=84, w1=None, h1=None, return_w1_h1=False):
	"""Vectorized CUDA implementation of random crop"""
	# assert isinstance(x, torch.Tensor) and x.is_cuda, 'input must be CUDA tensor'
	
	n = x.shape[0]
	img_size = x.shape[-1]
	crop_max = img_size - size

	if crop_max <= 0:
		if return_w1_h1:
			return x, None, None
		return x

	x = x.permute(0, 2, 3, 1)

	if w1 is None:
		w1 = torch.LongTensor(n).random_(0, crop_max)
		h1 = torch.LongTensor(n).random_(0, crop_max)

	windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
	cropped = windows[torch.arange(n), w1, h1]

	if return_w1_h1:
		return cropped, w1, h1

	return cropped


def view_as_windows_cuda(x, window_shape):
	"""PyTorch CUDA-enabled implementation of view_as_windows"""
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)

def add_random_boxes_cuda(x, boxes=15, mu=10, sigma=10, size_min=5, size_max=28, confuse_color=None, confuse_prob=0.0):
    """Vectorized CUDA implementation of random crop"""
    # assert isinstance(x, torch.Tensor) and x.is_cuda, 'input must be CUDA tensor'
    n = x.shape[0]
    img_size = x.shape[-1]

    x_perm = x.permute(0, 2, 3, 1)

    for _ in range(boxes):
        size = int(random.gauss(mu=mu, sigma=sigma))
        if size < size_min:
            size = size_min
        if size > size_max:
            size = size_max
        crop_max = img_size - size
        w1 = torch.LongTensor(n).random_(0, crop_max)
        h1 = torch.LongTensor(n).random_(0, crop_max)
        
        windows = view_as_windows_cuda(x_perm, (1, size, size, 1))[..., 0,:,:, 0]
        # n, 3, size, size
        if confuse_prob and random.random() < confuse_prob: # confuse_color: n, 3, 1, 1
            windows[torch.arange(n), w1, h1] = confuse_color + torch.randn_like(confuse_color) * 0.2
        else:
            windows[torch.arange(n), w1, h1] = torch.rand((n, 3, 1, 1), device=windows.device)
    
    return x

zero_images = []
kernel = np.ones((3,3),np.uint8)

def get_single_distractor(img_mask):
    img, mask = img_mask
    brightness = random.gauss(0.6, 0.2)
    #contrast = random.gauss(1, 0.05)
    #saturation = random.gauss(1, 0.05)
    img = TF.adjust_brightness(img, brightness)
    #img = TF.adjust_contrast(img, contrast)
    #img = TF.adjust_saturation(img, saturation)
    #if random.random() < 0.5:
    #    img = TF.hflip(img)
    # if random.random() < 0.5:
    img = TF.vflip(img)
    
    prev_device = img.device
    img = img.cpu()
    
    #opening = cv2.morphologyEx(mask.cpu().numpy(), cv2.MORPH_OPEN, kernel)
    
    erosion = cv2.erode(mask[0].cpu().numpy().astype(np.uint8),kernel,iterations = 1) # may lack a few pixels, but appear to be ok to extract
    if erosion.max() == 0:
        top_offset = random.random() * 50 + 20
        zero_images.append(img_mask)
    else:
        # 0 is height, 1 is width
        #print(erosion.nonzero()[1].max())
        k = erosion.nonzero()[0].max()
        top_offset = random.random() * 30 + 2 * k - img.shape[-2] + 5 # prev: 35
    #top_offset /= 2
    #print(erosion.nonzero()[0].max(), img.shape[-2], top_offset)
    left_offset = 0
    
    #plt.imshow(erosion)
    #plt.show()
    
    img = TF.to_pil_image(img)
    #img = Image.fromarray(erosion)
    img = TF.affine(img, angle=0, translate=(left_offset, top_offset), scale=1, shear=0)
    img = TF.to_tensor(img).to(device=prev_device)
    return img

def get_distracor(imgs, masks):
    imgs = apply_all(zip(imgs, masks), get_single_distractor)
    return torch.stack(imgs, dim=0)

def augment_background_with_data(x, dataset='places365_standard'):
    global places_iter
    alpha = torch.rand((x.size(0), 1, 1, 1), device=x.device) * (alpha_high - alpha_low) + alpha_low
    if dataset == 'places365_standard':
        imgs = _get_places_batch()
    else:
        raise NotImplementedError(f'overlay has not been implemented for dataset {dataset}')
    return (1-alpha)*x + (alpha)*imgs

def augment_frames(imgs):
    # Input: 9, 100, 100
    if weak_foreground_augment:
        hue = random.gauss(-0.001, 0.001)
        brightness = random.gauss(1, 0.01)
        contrast = random.gauss(1, 0.01)
        saturation = random.gauss(1, 0.01)
    else:
        hue = random.gauss(-0.015, 0.015)
        brightness = random.gauss(1, 0.08)
        contrast = random.gauss(1, 0.08)
        saturation = random.gauss(1, 0.05)
    
    img = imgs.view((3, 3, 100, 100)).cpu()
    
    img = apply_all(img, TF.to_pil_image)
    img = apply_all(img, TF.adjust_hue, hue)
    img = apply_all(img, TF.adjust_brightness, brightness)
    img = apply_all(img, TF.adjust_contrast, contrast)
    img = apply_all(img, TF.adjust_saturation, saturation)
    img = apply_all(img, TF.to_tensor)
    img = torch.cat(img, dim=0).to(device=config.device)
    return img

def crop_frames(imgs1, imgs2, masks): # -1, 9, 100, 100; -1, 9, 100, 100; -1, 3, 100, 100
    crop_top = random.randint(0, 16) # inclusive
    crop_left = random.randint(0, 16) # inclusive
    imgs_masks = torch.cat((imgs1, imgs2, masks), dim=1)
    cropped_imgs_masks = random_crop_cuda(imgs_masks)
    return cropped_imgs_masks[:, :9, ...], cropped_imgs_masks[:, 9:18, ...], cropped_imgs_masks[:, 18:21, ...]

def display_tensor_image(img, batch_idx=None):
    if not batch_idx is None:
        img = img[batch_idx]
    if img.shape[0] == 9:
        img = img[:3, ...]
    plt.imshow(img.permute([1, 2, 0]).cpu().detach().numpy().astype('float'))
    plt.show()

# dilate = False
    
from morphology import Dilation2d, Erosion2d
erosion = Erosion2d(in_channels=1, out_channels=1, kernel_size=3, soft_max=False).cuda()
dilation = Dilation2d(in_channels=1, out_channels=1, kernel_size=3, soft_max=False).cuda()

def generate_masks(x, foreground_threshold=0.49, display_image=False):
    with torch.no_grad():
        keypoints = model.point_net(x)
        features = model.feature_encoder(x)
        keypoints_gmap = transporter.gaussian_map(
                    transporter.spatial_softmax(keypoints), std=keypoint_std)
        masked = mask(keypoints_gmap, features)
        reconstruction2 = model.refine_net(masked) #reconstruction2: reconstructed from feature-based keypoints
        reconstruction_base = model.refine_net(torch.zeros_like(masked)[0:1])
        reconstruction2 = reconstruction2 - reconstruction_base

        if display_image:
            display_tensor_image(reconstruction2, 0)

        reconstruction_foreground = reconstruction2.max(dim=1, keepdim=True)[0] # select max
        if abs_comparison: # Some pixels may be darker than background pixels
            mask_foreground = torch.abs(reconstruction_foreground) > foreground_threshold
        else:
            mask_foreground = reconstruction_foreground > foreground_threshold
        
        if dilate:
            mask_foreground = dilation(mask_foreground.float())
            
        if erode:
            mask_foreground = erosion(mask_foreground.float())

        if display_image:
            plt.imshow(mask_foreground[0,0,:,:].cpu().detach().numpy().astype('float'))
            plt.show()

    return mask_foreground

# Background: don't need to be fully random, adding noise is enough
#masked_x_global = []
def generate_pairs(x, x_augmented, mask_foreground, display_image=False, random_confuse=True): # -1, 3, 84, 84; -1, 3, 84, 84; -1, 1, 84, 84
    with torch.no_grad():
        masked_x = x * mask_foreground
        #masked_x_global.append((masked_x, mask_foreground))
        
        masked_x_augmented = x_augmented * mask_foreground

        if display_image:
            display_tensor_image(masked_x_augmented, 0)

        noise = torch.rand_like(x_augmented)
        if not random_confuse:
            random_background = noise # different background for each of three frames
        else:
            background_randomness = torch.rand((x.shape[0], 1, 1, 1), device=config.device)
            # Actually, the background may need to be consistent if we apply to 3 images.
            
            # If random_confuse for instance, give 0.3 noise, 0.4 confusion.
            
            # 0-0.3: original bg, noise
            # 0.3-0.7: random bg, noise
            # 0.7-1.0: color bg, noise
            
            confuse_color = masked_x_augmented.view((-1, 3, 84 * 84)).sum(dim=2) \
                / (mask_foreground.view((-1, 1, 84 * 84)).sum(dim=2) + 1e-5)
            confuse_color = confuse_color.view((-1, 3, 1, 1))
            
            random_color = torch.rand((x_augmented.shape[0], 3, 1, 1), device=config.device)
            
            original_bg = background_randomness < 0.3
            random_bg = (background_randomness >= 0.3) & (background_randomness < 0.8)
            color_bg = background_randomness >= 0.8
            
            if distracor_probability > 0:
                distractor_randomness = torch.rand((x.shape[0], 1, 1, 1), device=config.device)
                distractor_bg = distractor_randomness >= (1 - distracor_probability)
                distactor = get_distracor(masked_x, mask_foreground)
            
            bg_percentage = 0.5
            
            random_background = noise * noise_level + \
                x_augmented * (original_bg * bg_percentage) + \
                random_color * (random_bg * bg_percentage) + \
                (confuse_color + torch.randn_like(confuse_color) * 0.2) * (color_bg * bg_percentage)
            
            if add_random_boxes:
                add_random_boxes_cuda(random_background, **boxes_kwargs)
            
            if distracor_probability > 0:
                random_background = random_background * (1 - distractor_bg * 0.3) + distactor * (distractor_bg * 0.6)
            
            random_background = random_background.clamp(min=0, max=1)
            
            random_background = augment_background_with_data(random_background)
            
        if display_image:
            display_tensor_image(random_background, 0)

        noisy_masked_x_augmented = masked_x_augmented + random_background * (1 - mask_foreground)

        if display_image:
            display_tensor_image(noisy_masked_x_augmented, 0)
        
        # noisy_masked_x_augmented = x_augmented
    
    return masked_x, noisy_masked_x_augmented

no_aug = False # This is for ablation studies.
no_aug_with_original_image = False
if no_aug:
    print("No aug is True")

def run():
    _reset_places()
    encoder, decoder = _get_adapter_model(config)
    encoder.train()
    decoder.train()
    loader = _get_data_loader(config)
    intermediate_factor = 0.5
    display_image = False
    display_output = False
    show_images = False
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        config.learning_rate_decay_every_n_steps,
        gamma=config.learning_rate_decay_rate)
    
    for itr, x_9_view in enumerate(loader):
        # For display purpose
        #x_9_view = images
        x_9_view = x_9_view.to(device=config.device).float()
        # x_9_view = x_9_view[:16]
        # print(itr, "x_9_view:", x_9_view.shape)
        x_3_view = x_9_view.view((-1, 3, 100, 100))
        
        mask_foreground = generate_masks(x_3_view, foreground_threshold=foreground_threshold, display_image=display_image).view((-1, 3, 100, 100))
        
        x_augmented = torch.stack(
            [augment_frames(x_item) for x_item in x_9_view],
            dim=0) # same augmentation for 3 images, shape: (-1, 9, 100, 100)
        
        # print("x_augmented", x_augmented.shape)
        
        x_cropped, x_augmented_cropped, masks_cropped = crop_frames(x_9_view, x_augmented, mask_foreground)
        
        x_cropped, x_augmented_cropped, masks_cropped = x_cropped.reshape((-1, 3, 84, 84)), \
            x_augmented_cropped.reshape((-1, 3, 84, 84)), masks_cropped.reshape((-1, 1, 84, 84))
        
        # print("x_augmented_cropped", x_augmented_cropped.shape)
        # print("masks_cropped", masks_cropped.shape)
        masked_image, noisy_image = generate_pairs(x_cropped, x_augmented_cropped, masks_cropped, display_image=display_image)
        
        # detect problematic masks
        # zero_mask = masks_cropped.view((masks_cropped.size(0), -1)).sum(dim=1) == 0
        #if torch.any(zero_mask):
        #    print("Detect problematic mask. Number: ", zero_mask.sum().item())

        # Comment out to pass the image separately
        #masked_image = masked_image.view((-1, 9, 84, 84))
        #noisy_image = noisy_image.view((-1, 9, 84, 84))
        
        if display_output:
            for ind in range(30):
                # print(x_cropped.shape, noisy_image.shape)
                plt.axis("off")
                display_tensor_image(x_cropped[ind, :3, ...])
                
                #plt.axis("off")
                #plt.imshow(masks_cropped[ind, 0, ...].cpu().detach().numpy().astype('float'), cmap="gray")
                #plt.show()
                
                #plt.axis("off")
                #display_tensor_image(masked_image[ind, :3, ...])
                plt.axis("off")
                display_tensor_image(noisy_image[ind, :3, ...])

                #display_tensor_image(masked_image[ind, 3:6, ...])
                #display_tensor_image(noisy_image[ind, 3:6, ...])

                #display_tensor_image(masked_image[ind, 6:, ...])
                #display_tensor_image(noisy_image[ind, 6:, ...])
            return
        
        #assert masked_image.shape[0] == noisy_image.shape[0] == x_9_view.shape[0]
        
        optimizer.zero_grad()
        if no_aug:
            if no_aug_with_original_image:
                intermediate = encoder(x_cropped)
            else:
                intermediate = encoder(masked_image)
        else:
            intermediate = encoder(noisy_image)
        intermediate_masked = encoder(masked_image).detach() # add this loss
        reconstruction = decoder(intermediate).clamp(min=0, max=1)
        
        if neg_weight == 1.0 and pos_weight == 1.0:
            reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, masks_cropped)
        else: # weighted
            reconstruction_loss = torch.mean(neg_weight * F.relu(reconstruction - masks_cropped) ** 2 + \
                               pos_weight * F.relu(masks_cropped - reconstruction) ** 2)
        
        if no_aug and (not no_aug_with_original_image):
            intermediate_loss = reconstruction_loss * 0
        else:
            intermediate_loss = nn.functional.mse_loss(intermediate, intermediate_masked)
        
        loss = reconstruction_loss + intermediate_loss * intermediate_factor
        loss.backward()

        optimizer.step()
        scheduler.step()
        
        # print(reconstruction.shape)
        
        if itr % config.report_every_n_steps == 0:
            print('Itr ', itr, 'Loss ', reconstruction_loss.item(), intermediate_loss.item())

            if torch.isnan(reconstruction_loss):
                print("Encounter a NaN")
                return
            
            torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, os.path.join(config.log_file))

            if show_images and itr % 200 == 0:
                masked_image = torchvision.utils.make_grid(masked_image, nrow=16)
                noisy_image = torchvision.utils.make_grid(noisy_image, nrow=16)
                
                reconst_grid = torchvision.utils.make_grid(reconstruction * x_cropped, nrow=16)

                plt.figure(figsize=(10,10))
                plt.imshow(np.transpose(masked_image[:3].detach().cpu(), (1,2,0)))
                plt.axis("off")
                plt.show()
                
                plt.figure(figsize=(10,10))
                plt.imshow(np.transpose(noisy_image[:3].detach().cpu(), (1,2,0)))
                plt.axis("off")
                plt.show()
                
                plt.figure(figsize=(10,10))
                plt.imshow(np.transpose(reconst_grid[:3].detach().cpu(), (1,2,0)))
                plt.axis("off")
                plt.show()
            if itr > config.num_iterations:
                break
run()

