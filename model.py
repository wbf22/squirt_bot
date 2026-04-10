import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


"""
Loss function — since you asked earlier, for binary mask output the standard is binary cross entropy + dice loss combined. 
BCE handles per-pixel accuracy, dice loss handles the class imbalance problem (most pixels are background).
"""


KERNAL_SIZE = 3
STRIDE = 2
PADDING = 1
EMBED_DIM = 256
EMBED_DIM_HALF = EMBED_DIM//2
class TargetDetector(nn.Module):

    def __init__(self):

        """
        CONVOLUTIONS
        - image is converted into useful features
        - target example images are converted into useful features
        """
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 32, KERNAL_SIZE, 1, PADDING, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, KERNAL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, KERNAL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.residual_0 = nn.Sequential(
            nn.Conv2d(128, 64, 1, STRIDE, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, KERNAL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.residual_1 = nn.Sequential(
            nn.Conv2d(256, 128, 1, STRIDE, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.residual_2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, STRIDE, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, KERNAL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.residual_3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, STRIDE, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.residual_4 = nn.Sequential(
            nn.Conv2d(512, 256, 1, STRIDE, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.residual_5 = nn.Sequential(
            nn.Conv2d(512, 256, 1, STRIDE, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, KERNAL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.residual_6 = nn.Sequential(
            nn.Conv2d(512, 256, 1, STRIDE, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(512, EMBED_DIM_HALF, 1, STRIDE, 0, bias=False),
            nn.BatchNorm2d(EMBED_DIM_HALF),
            nn.ReLU(inplace=True)
        )

        self.lat_p3 = nn.Sequential(
            nn.Conv2d(256, EMBED_DIM_HALF, 1, STRIDE, 0, bias=False),
            nn.BatchNorm2d(EMBED_DIM_HALF),
            nn.ReLU(inplace=True)
        )

        """
        ATTENTION
        - divide images into overlapping patches
        - each patch contains both high res and low res convolutional output
        - attend each example image patch to each patch in image
        - query is from example images, key and value come from image
        """
        self.pos_embed = nn.Parameter(torch.randn(1, 0, EMBED_DIM)) # a learned parameter that is added to each patch. [1, W*H, EMBED_DIM] so each xy posiiton gets a unique value
        self.mha = nn.MultiheadAttention(
            embed_dim=256,   # your C dimension
            num_heads=8,
            batch_first=True # makes input [B, seq_len, C] instead of [seq_len, B, C]
        )

        """
        MASK CREATION
        - for attention output patches, merge into image again with averaging where overlapping
        - deconvolude up to a mask
        - sigmoid the output of that for 0-1
        """

    def convolutions(
        self,
        x # [BATCH, 1920, 1080, 3]
    ):
        # CONVOLUTIONS
        x = self.conv_0(x)
        x = self.conv_1(x)

        x = self.conv_2(x)
        x = x + self.residual_0(x)
        
        x = self.conv_3(x)
        x = x + self.residual_1(x)
        x = x + self.residual_2(x)
        p3 = x

        x = self.conv_4(x)
        x = x + self.residual_3(x)
        x = x + self.residual_4(x)
        x = x + self.residual_5(x)

        x = self.conv_5(x)
        x = x + self.residual_6(x)
        x = self.conv_6(x)
        p5 = x # [BATCH, W, H, EMBED_DIM_HALF]

        p3 = self.lat_p3(p3) # [BATCH, W, H, EMBED_DIM_HALF]

        return torch.cat([p5, p3], dim=3)

    def forward(
        self, 
        input_image, # [BATCH, 1920, 1080, 3]
        queries # [BATCH, CONV_WIDTH*CONV_HEIGHT*num_examples, EMBED_DIM] (result of convolutions, the concatted in dim 1 with other examples)
    ):

        # CONVOLUTIONS
        x = self.convolutions(input_image)

        # ATTENTION

        # Cross attention 
        out, attn_weights = self.mha(query=queries, key=x, value=x)

        

