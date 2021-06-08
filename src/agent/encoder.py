import torch
import torch.nn as nn
from utils import add_random_boxes_cuda

import warnings

OUT_DIM = {2: 39, 4: 35, 6: 31, 8: 27, 10: 23, 11: 21, 12: 19}
all_encoders = []

def tie_weights(src, trg):
	assert type(src) == type(trg)
	trg.weight = src.weight
	trg.bias = src.bias


class CenterCrop(nn.Module):
	"""Center-crop if observation is not already cropped"""
	def __init__(self, size):
		super().__init__()
		assert size == 84
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		elif x.size(-1) == 100:
			return x[:, :, 8:-8, 8:-8]
		else:
			return ValueError('unexepcted input size')


class NormalizeImg(nn.Module):
	"""Normalize observation"""
	def forward(self, x):
		return x/255.

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1,
                 padding=1):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)

class AdapterEncoder(nn.Module):
    def __init__(self, obs_shape, num_filters, num_in_filters):
        super().__init__()
        
        self.front = nn.Sequential(
            Block(obs_shape[0], num_in_filters, kernel_size=(3, 3), stride=2, padding=1), # 1
            Block(num_in_filters, num_filters, kernel_size=(3, 3), stride=1)  # 2
        )
        
        self.attention = Block(num_filters, num_filters, kernel_size=(3, 3), stride=2) # 3
        self.middle = Block(num_filters, num_filters, kernel_size=(3, 3), stride=2) # 3
        
        self.last = Block(num_filters, num_filters, kernel_size=(3, 3), stride=1)  # 4
        
    def forward(self, x):
        after_front = self.front(x)
        
        after_attention = self.attention(after_front).sigmoid()
        after_middle = self.middle(after_front)
        
        # print(after_front.shape, after_attention.shape, after_middle.shape)
        return self.last(after_middle * after_attention) # attention will broadcast in channels

def get_encoder():
	obs_shape = (3, 84, 84)
	num_filters = 16
	num_in_filters = 16
	encoder = AdapterEncoder(obs_shape, num_filters, num_in_filters).cuda()
	encoder.eval()
	return encoder

def get_decoder():
	obs_shape = (3, 84, 84)
	num_filters = 16
	decoder = nn.Sequential(
        Block(num_filters, num_filters, kernel_size=(3, 3), stride=1),  # 4
        Block(num_filters, num_filters, kernel_size=(3, 3), stride=1),  # 3
        nn.UpsamplingBilinear2d(scale_factor=2),
        Block(num_filters, num_filters, kernel_size=(3, 3), stride=1),  # 2
        Block(num_filters, 1, kernel_size=(3, 3), stride=1, padding=1), # 1
        nn.UpsamplingBilinear2d(scale_factor=2)
    )
	decoder = decoder.cuda()
	decoder.eval()
	return decoder

image_encoder = get_encoder()
image_decoder = get_decoder()

count = [0]

class AdaptObservation(nn.Module):
	def __init__(self, adapt_aug=None):
		super().__init__()
		self.adapt_aug = adapt_aug
		print("Adapt aug: %s (Please make augmentations weaker if you encounter instability or difficulty in learning in training environment and make it stronger if you encounter instability when adapter misclassifies a few patches)"%adapt_aug)

	def forward(self, x):
		image_encoder.eval()
		image_decoder.eval()
		with torch.no_grad():
			if x.shape[1] == 18:
				x, un_denoised_x = x[:, :x.shape[1] // 2, ...], x[:, x.shape[1] // 2:, ...]
			elif x.shape[1] == 9:
				# print("Warning: Using un-denoised input")
				un_denoised_x = x
			else:
				assert False, "Unsupported observation shape: %s" % x.shape

			x_3_view = x.view((-1, 3, 84, 84))
			x_un_denoised_3_view = un_denoised_x.view((-1, 3, 84, 84))
			encoded = image_encoder(x_3_view)
			decoded = image_decoder(encoded)
			if False:
				warnings.warn("Clamp should be enabled only for finger, turn_easy")
				# Original statement: (division does not take effect)
				# decoded.clamp_(min=0, max=0.8) / 0.8
				decoded = decoded.clamp_(min=0, max=0.8) / 0.8
			else:
				warnings.warn("No clamp to smaller values: clamp should be enabled only for finger, turn_easy")
				decoded = decoded.clamp_(min=0, max=1.0)
			# decoded = (decoded * x_un_denoised_3_view)
			# Use contiguous() if you encounter problems
			decoded = (decoded * x_un_denoised_3_view).contiguous()
			
			return decoded.view((-1, 9, 84, 84))

def adapter_load_state_dict(state_dict):
	image_encoder.load_state_dict(state_dict["encoder"])
	image_decoder.load_state_dict(state_dict["decoder"])
	for param in image_encoder.parameters():
		param.requires_grad_(False)
	for param in image_decoder.parameters():
		param.requires_grad_(False)

class PixelEncoder(nn.Module):
	"""Convolutional encoder of pixel observations"""
	def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4, adapt_aug=None, adapt_observation=True):
		super().__init__()
		assert len(obs_shape) == 3

		self.feature_dim = feature_dim
		self.num_layers = num_layers
		self.num_shared_layers = num_shared_layers
		self.adapt_aug = adapt_aug
		if adapt_observation:
			print("Adapt observation")
			self.preprocess = nn.Sequential(
				CenterCrop(size=84), NormalizeImg(), AdaptObservation(adapt_aug=adapt_aug)
			)
		else:
			print("Do not adapt observation")
			self.preprocess = nn.Sequential(
				CenterCrop(size=84), NormalizeImg()
			)

		self.convs = nn.ModuleList(
			[nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		)
		for i in range(num_layers - 1):
			self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

		out_dim = OUT_DIM[num_layers]
		self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
		self.ln = nn.LayerNorm(self.feature_dim)

	def forward_conv(self, obs, detach=False):
		if False and count[0] % 20 == 0:
			# image_encoder.front[0].conv.weight[0], image_encoder.front[0].bn.state_dict()
			# image_decoder[0].conv.weight[0], image_decoder[0].bn.state_dict()
			obs_original = obs
			torch.save(obs_original, "obs_original.pth", _use_new_zipfile_serialization=False)
			obs_processed = self.preprocess(obs_original) # Note: this one is normalized
			torch.save(obs_processed, "obs_processed.pth", _use_new_zipfile_serialization=False)
			import ipdb; ipdb.set_trace()
		

		obs = self.preprocess(obs)
		if self.adapt_aug:
			obs = obs.view((-1, 3, 84, 84))
			add_random_boxes_cuda(obs, boxes=15, mu=0, sigma=2, size_min=1, size_max=5)
			obs += torch.randn_like(obs) * 0.005
			obs = obs.view((-1, 9, 84, 84))
		conv = torch.relu(self.convs[0](obs))

		if False and count[0] % 20 == 0:
			torch.save(obs, "obs_processed_aug.pth", _use_new_zipfile_serialization=False)
			import ipdb; ipdb.set_trace()

		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			if i == self.num_shared_layers-1 and detach:
				conv = conv.detach()
		count[0] += 1
		h = conv.view(conv.size(0), -1)
		return h

	def forward(self, obs, detach=False):
		h = self.forward_conv(obs, detach)
		h_fc = self.fc(h)
		h_norm = self.ln(h_fc)
		out = torch.tanh(h_norm)

		return out

	def copy_conv_weights_from(self, source, n=None):
		"""Tie n first convolutional layers"""
		if n is None:
			n = self.num_layers
		for i in range(n):
			tie_weights(src=source.convs[i], trg=self.convs[i])


def make_encoder(
	obs_shape, feature_dim, num_layers, num_filters, num_shared_layers, adapt_aug, adapt_observation
):
	assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
	if num_shared_layers == -1 or num_shared_layers == None:
		num_shared_layers = num_layers
	assert num_shared_layers <= num_layers and num_shared_layers > 0, \
		f'invalid number of shared layers, received {num_shared_layers} layers'
	encoder = PixelEncoder(
		obs_shape, feature_dim, num_layers, num_filters, num_shared_layers, adapt_aug=adapt_aug, adapt_observation=adapt_observation
	)
	all_encoders.append(encoder)
	return encoder
