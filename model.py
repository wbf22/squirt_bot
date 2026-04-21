import math
import torch
import torch.nn as nn
import torch.nn.functional as F


KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
EMBED_DIM = 256
HEIGHT = 1080
WIDTH = 1920


class TargetDetector(nn.Module):
    def __init__(self):

        """
        So we'll feed in do and don't images to convolutions

        subtract the embeddings from each other to get the semantic difference.
        feed that into the model with the do embedding
        """



        """
        CONVOLUTIONS
        - image is converted into useful features
        - target example images are converted into useful features
        """
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 32, KERNEL_SIZE, 1, PADDING, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(32, 64, KERNEL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, KERNEL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.residual_0 = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, PADDING, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, KERNEL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.residual_1 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, PADDING, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.residual_2 = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, PADDING, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, KERNEL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.residual_3 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.residual_4 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.residual_5 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 512, KERNEL_SIZE, STRIDE, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.residual_6 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 1, PADDING, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(512, EMBED_DIM, 1, 1, 0, bias=False),
            nn.BatchNorm2d(EMBED_DIM),
            nn.ReLU(inplace=True),
        )

        # lateral from p3
        self.lat_p3 = nn.Sequential(
            nn.Conv2d(256, EMBED_DIM, 1, 1, 0, bias=False),
            nn.BatchNorm2d(EMBED_DIM),
            nn.ReLU(inplace=True),
        )

        # projection to merge p5+p3 (they are concatenated -> 2*EMBED_DIM -> EMBED_DIM)
        self.merge_proj = nn.Sequential(
            nn.Conv2d(EMBED_DIM * 2, EMBED_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(EMBED_DIM),
            nn.ReLU(inplace=True),
        )
        """
        ATTENTION
        - divide images into overlapping patches
        - each patch contains both high res and low res convolutional output
        - attend each example image patch to each patch in image
        - query is from example images, key and value come from image
        """
        self.pos_embed = None  # optional learned positional embeddings (set externally if desired)
        self.mha = nn.MultiheadAttention(embed_dim=EMBED_DIM, num_heads=8, batch_first=True)


        """
        MASK CREATION
        - for attention output patches, merge into image again with averaging where overlapping
        - deconvolude up to a mask
        - sigmoid the output of that for 0-1
        """
        self.mask_head = nn.Sequential(
            nn.Conv2d(EMBED_DIM, EMBED_DIM // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(EMBED_DIM // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(EMBED_DIM // 2, 1, kernel_size=1),
        )

        # store last spatial dims when convolutions() is called

    def convolutions(self, x):
        """
        x: [BATCH, 3, H, W]
        returns: seq [BATCH, H_feat*W_feat, EMBED_DIM]
        and stores last spatial dims in self._last_hw
        """
        x = self.conv_0(x)
        x = self.conv_1(x)

        x = self.conv_2(x)
        x = x + self.residual_0(x)

        x = self.conv_3(x)
        x = x + self.residual_1(x)
        x = x + self.residual_2(x)
        p3 = x  # keep for lateral

        x = self.conv_4(x)
        x = x + self.residual_3(x)
        x = x + self.residual_4(x)
        x = x + self.residual_5(x)

        x = self.conv_5(x)
        x = x + self.residual_6(x)
        x = self.conv_6(x)
        p5 = x

        p3 = self.lat_p3(p3)

        # ensure p5 and p3 have same spatial dims (upsample p5 to p3)
        if p5.shape[2:] != p3.shape[2:]:
            p5 = F.interpolate(p5, size=p3.shape[2:], mode='bilinear', align_corners=False)

        feat = torch.cat([p5, p3], dim=1)  # [BATCH, 2*EMBED_DIM, H_feat, W_feat]
        # project back to EMBED_DIM so sequence embeddings match attention embed_dim
        feat = self.merge_proj(feat)
        seq = feat.view(feat.size(0), EMBED_DIM, HEIGHT * WIDTH).permute(0, 2, 1).contiguous()  # [BATCH, HEIGHT*WIDTH, EMBED_DIM]
        return seq

    def forward(self, input_image, examples, anti_examples):
        """
        input_image: either raw image tensor [BATCH, 3, H, W] 
        examples: expected [BATCH, examples, HEIGHT*WIDTH, S, EMBED_DIM]
        anti_examples: [BATCH, anti_examples, HEIGHT*WIDTH, S, EMBED_DIM]

        returns: mask [BATCH,1,H_out,W_out]
        """
        x = self.convolutions(input_image)  # [BATCH, HEIGHT*WIDTH, EMBED_DIM]

        # compute diff and keys/values
        diff = examples - anti_examples
        key = torch.cat([examples, diff], dim=1)  # [BATCH, n, HEIGHT*WIDTH, EMBED_DIM]

        # multi-head attention (queries from image sequence)
        out, attn_weights = self.mha(query=x, key=key, value=key)  # out: [BATCH, HEIGHT*WIDTH, EMBED_DIM]

        # reshape attention output to spatial
        feat = out.permute(0, 2, 1).contiguous().view(out.size(0), EMBED_DIM, HEIGHT, WIDTH)  # [BATCH, EMBED_DIM, H, W]

        mask_logits = self.mask_head(feat)  # [BATCH,1,H,W]

        mask = torch.sigmoid(mask_logits)
        return mask, attn_weights
