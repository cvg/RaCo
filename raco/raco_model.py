import torch
import torch.nn as nn
import torch.nn.functional as F


class InputPadder(object):
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, h: int, w: int, divis_by: int = 8):
        self.ht = h
        self.wd = w
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]

    def pad(self, x: torch.Tensor):
        assert x.ndim == 4
        return F.pad(x, self._pad, mode="replicate")

    def unpad(self, x: torch.Tensor):
        assert x.ndim == 4
        ht = x.shape[-2]
        wd = x.shape[-1]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.gate = nn.SELU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.gate(self.bn1(self.conv1(x)))  # B x in_channels x H x W
        x = self.gate(self.bn2(self.conv2(x)))  # B x out_channels x H x W
        return x


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
    ) -> None:
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.gate = nn.SELU(inplace=True)
        self.match_dims = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gate(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.match_dims(identity)

        out += identity
        out = self.gate(out)

        return out


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, kernel_size=3):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=False,
    )


class RacoModel(nn.Module):  # Raco multiscale architecture with standard convolutions
    # Modified version of ALIKED-n16 https://github.com/Shiaoming/ALIKED
    
    default_conf = {
        "name": "raco_model",
        "ranker": True,
        "covariance_estimator": True,
    }

    def __init__(self, conf):
        super().__init__()

        conf = {**self.default_conf, **conf}
        self.run_ranker = conf["ranker"]
        self.run_covariance_estimator = conf["covariance_estimator"]

        # ALIKED-n16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d
        self.gate = nn.SELU(inplace=True)
        c1 = 16
        c2 = 32
        c3 = 64
        c4 = 128
        self.block1 = ConvBlock(3, c1)
        self.block2 = ResBlock(c1, c2)
        self.block3 = ResBlock(c2, c3)
        self.block4 = ResBlock(c3, c4)
        dim = c4

        self.conv1 = conv1x1(c1, dim // 4)
        self.conv2 = conv3x3(c2, dim // 4)
        self.conv3 = conv3x3(c3, dim // 4)
        self.conv4 = conv3x3(c4, dim // 4)
        self.upsample2 = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample4 = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=True
        )
        self.upsample8 = nn.Upsample(
            scale_factor=8, mode="bilinear", align_corners=True
        )
        self.upsample32 = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=True
        )
        self.score_head = nn.Sequential(
            conv1x1(dim, 8),
            self.gate,
            conv3x3(8, 4),
            self.gate,
            conv3x3(4, 4),
            self.gate,
            conv3x3(4, 1),
        )

        # Use a loop to create multiple ResBlocks for the ranker_head
        ranker_layers = [ResBlock(3, 12)]
        ranker_layers += [ResBlock(12, 12) for _ in range(8)]
        ranker_layers += [
            nn.Conv2d(
                12, 1, kernel_size=5, padding=2, bias=True, padding_mode="reflect"
            )
        ]
        self.ranker_head = nn.Sequential(*ranker_layers)

        modules = []
        in_channels = dim
        for out_channels in [64, 32, 32]:
            modules.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                    padding_mode="reflect",
                )
            )
            modules.append(nn.LeakyReLU(inplace=True))
            in_channels = out_channels
        modules.append(
            nn.Conv2d(
                in_channels=32,
                out_channels=3,
                kernel_size=1,
                bias=True,
                padding_mode="reflect",
            )
        )
        self.covariance_estimator_head = nn.Sequential(*modules)

    def forward(self, x):
        assert x.shape[1] == 3, "Input must be RGB image with 3 channels"
        div_by = 2**5
        padder = InputPadder(x.shape[-2], x.shape[-1], div_by)
        x = padder.pad(x)

        # Feature extraction
        x1 = self.block1(x)  # B x c1 x H x W
        x2 = self.pool2(x1)
        x2 = self.block2(x2)  # B x c2 x H/2 x W/2
        x3 = self.pool4(x2)
        x3 = self.block3(x3)  # B x c3 x H/8 x W/8
        x4 = self.pool4(x3)
        x4 = self.block4(x4)  # B x dim x H/32 x W/32

        # Feature aggregation
        x1 = self.gate(self.conv1(x1))  # B x dim//4 x H x W
        x2 = self.gate(self.conv2(x2))  # B x dim//4 x H//2 x W//2
        x3 = self.gate(self.conv3(x3))  # B x dim//4 x H//8 x W//8
        x4 = self.gate(self.conv4(x4))  # B x dim//4 x H//32 x W//32
        x2_up = self.upsample2(x2)  # B x dim//4 x H x W
        x3_up = self.upsample8(x3)  # B x dim//4 x H x W
        x4_up = self.upsample32(x4)  # B x dim//4 x H x W
        x1234 = torch.cat([x1, x2_up, x3_up, x4_up], dim=1)

        # TODO(Abhiram): only for training subpixel covariances!
        # detached_x1234 = x1234.detach()
        detached_x1234 = x1234

        # score head
        score_map = self.score_head(x1234)
        score_map = padder.unpad(score_map)
        # B x 1 x H x W

        ranker_map, cov_maps = None, None
        if self.run_ranker:
            ranker_map = self.ranker_head(x)
            ranker_map = padder.unpad(ranker_map)
            # B x 1 x H x W

        if self.run_covariance_estimator:
            cov_maps = self.covariance_estimator_head(detached_x1234)
            cov_maps = padder.unpad(cov_maps)
            # B x 3 x H x W

        return score_map, ranker_map, cov_maps
