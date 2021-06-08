import torch
from torch import nn
from utils import spatial_softmax


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


def get_encoder(in_channels, return_net_channels=False):
    net = nn.Sequential(
            Block(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1), # 1
            Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            Block(32, 64, kernel_size=(3, 3), stride=2),  # 3
            Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            Block(64, 64, kernel_size=(3, 3), stride=2), # 5
            Block(64, 64, kernel_size=(3, 3), stride=1)  # 6
        )

    if return_net_channels:
        return net, 64
    return net

def get_decoder(num_channels):
    return nn.Sequential(
            Block(64, 64, kernel_size=(3, 3), stride=1), # 6 
            Block(64, 64, kernel_size=(3, 3), stride=1), # 5
            nn.UpsamplingBilinear2d(scale_factor=2),
            Block(64, 64, kernel_size=(3, 3), stride=1),  # 4
            Block(64, 32, kernel_size=(3, 3), stride=1),  # 3
            nn.UpsamplingBilinear2d(scale_factor=2),
            Block(32, 32, kernel_size=(3, 3), stride=1),  # 2
            Block(32, num_channels, kernel_size=(3, 3), stride=1, padding=1) # 1
        )

class FeatureEncoder(nn.Module):
    """Phi"""

    def __init__(self, in_channels=3):
        super(FeatureEncoder, self).__init__()
        self.net = get_encoder(in_channels)


    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.

        Returns
        =======
        y: (N, C, H, K) tensor.
        """
        return self.net(x)


class PoseRegressor(nn.Module):
    """Pose regressor"""

    # https://papers.nips.cc/paper/7657-unsupervised-learning-of-object-landmarks-through-conditional-image-generation.pdf

    def __init__(self, in_channels=3, k=1,):
        super(PoseRegressor, self).__init__()
        self.net, net_channels = get_encoder(in_channels, return_net_channels=True)
        self.regressor = nn.Conv2d(net_channels, k, kernel_size=(1, 1))

    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.
        
        Returns
        =======
        y: (N, k, H', W') tensor.
        """
        x = self.net(x)
        return self.regressor(x)


class RefineNet(nn.Module):
    """Network that generates images from feature maps and heatmaps."""

    def __init__(self, num_channels):
        super(RefineNet, self).__init__()
        self.net = get_decoder(num_channels)

    def forward(self, x):
        """
        x: the transported feature map.
        """
        return self.net(x)

def compute_keypoint_location_mean(features):
    S_row = features.sum(-1)  # N, K, H
    S_col = features.sum(-2)  # N, K, W

    # N, K
    u_row = S_row.mul(torch.linspace(-1, 1, S_row.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    # N, K
    u_col = S_col.mul(torch.linspace(-1, 1, S_col.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    return torch.stack((u_row, u_col), -1) # N, K, 2


def gaussian_map(features, std=0.2):
    # features: (N, K, H, W)
    width, height = features.size(-1), features.size(-2)
    mu = compute_keypoint_location_mean(features)  # N, K, 2
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = torch.linspace(-1.0, 1.0, height, dtype=mu.dtype, device=mu.device)
    x = torch.linspace(-1.0, 1.0, width, dtype=mu.dtype, device=mu.device)
    mu_y, mu_x = mu_y.unsqueeze(-1), mu_x.unsqueeze(-1)

    y = torch.reshape(y, [1, 1, height, 1])
    x = torch.reshape(x, [1, 1, 1, width])

    inv_std = 1 / std
    g_y = torch.pow(y - mu_y, 2)
    g_x = torch.pow(x - mu_x, 2)
    dist = (g_y + g_x) * inv_std**2
    g_yx = torch.exp(-dist)
    # g_yx = g_yx.permute([0, 2, 3, 1])
    return g_yx


def transport(source_keypoints, target_keypoints, source_features,
              target_features):
    """
    Args
    ====
    source_keypoints (N, K, H, W)
    target_keypoints (N, K, H, W)
    source_features (N, D, H, W)
    target_features (N, D, H, W)

    Returns
    =======
    """
    out = source_features
    for s, t in zip(torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)):
        out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * out + t.unsqueeze(1) * target_features
    return out


class Transporter(nn.Module):

    def __init__(self, feature_encoder, point_net, refine_net, std=0.1):
        super(Transporter, self).__init__()
        self.feature_encoder = feature_encoder
        self.point_net = point_net
        self.refine_net = refine_net
        self.std = std

    def forward(self, source_images, target_images):
        source_features = self.feature_encoder(source_images)
        target_features = self.feature_encoder(target_images)

        source_keypoints = gaussian_map(
            spatial_softmax(self.point_net(source_images)), std=self.std)

        target_keypoints = gaussian_map(
            spatial_softmax(self.point_net(target_images)), std=self.std)

        transported_features = transport(source_keypoints.detach(),
                                         target_keypoints,
                                         source_features.detach(),
                                         target_features)

        assert transported_features.shape == target_features.shape

        reconstruction = self.refine_net(transported_features)
        return reconstruction
