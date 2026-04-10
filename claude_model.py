import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Conv2D → BatchNorm → ReLU block (the basic unit of the backbone)."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block with a skip connection to prevent vanishing gradients."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(channels, channels // 2, kernel_size=1, padding=0),
            ConvBNReLU(channels // 2, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection


# ---------------------------------------------------------------------------
# Backbone — Feature Extraction
# ---------------------------------------------------------------------------
# Progressively downsamples the image through 5 stages, producing feature
# maps at 3 different scales (large, medium, small) for the neck.

class Backbone(nn.Module):
    """
    Darknet-inspired backbone.

    Input : (B, 3, H, W)
    Output: three feature maps at strides 8, 16, 32
              - P3: (B, 128, H/8,  W/8)   ← small objects
              - P4: (B, 256, H/16, W/16)  ← medium objects
              - P5: (B, 512, H/32, W/32)  ← large objects
    """

    def __init__(self):
        super().__init__()

        # Stage 0 – initial stem: full resolution → /2
        self.stage0 = nn.Sequential(
            ConvBNReLU(3, 32, kernel_size=3),
            ConvBNReLU(32, 64, stride=2)   # /2
        )

        # Stage 1 – /4
        self.stage1 = nn.Sequential(
            ConvBNReLU(64, 128, stride=2),  # /4
            ResidualBlock(128),
        )

        # Stage 2 – /8  →  P3
        self.stage2 = nn.Sequential(
            ConvBNReLU(128, 256, stride=2),  # /8
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # Stage 3 – /16  →  P4
        self.stage3 = nn.Sequential(
            ConvBNReLU(256, 512, stride=2),  # /16
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
        )

        # Stage 4 – /32  →  P5
        self.stage4 = nn.Sequential(
            ConvBNReLU(512, 1024, stride=2),  # /32
            ResidualBlock(1024),
            ConvBNReLU(1024, 512, kernel_size=1, padding=0),  # channel squeeze
        )

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # stride 8,  128 ch  (will reduce to 128 in neck)
        p4 = self.stage3(p3)  # stride 16, 512 ch  (will reduce to 256 in neck)
        p5 = self.stage4(p4)  # stride 32, 512 ch
        return p3, p4, p5


# ---------------------------------------------------------------------------
# Neck — Feature Pyramid Network (FPN)
# ---------------------------------------------------------------------------
# Merges deep (semantic) features back up with shallow (spatial) features so
# that the detection head can detect objects at every scale.

class FPN(nn.Module):
    """
    Top-down FPN neck.

    Inputs : P3 (128ch), P4 (512ch), P5 (512ch)
    Outputs: N3 (128ch), N4 (128ch), N5 (128ch)  — all 128-channel
    """

    def __init__(self):
        super().__init__()

        # Lateral 1×1 convolutions to align channel counts
        self.lat_p5 = ConvBNReLU(512, 128, kernel_size=1, padding=0)
        self.lat_p4 = ConvBNReLU(512, 128, kernel_size=1, padding=0)
        self.lat_p3 = ConvBNReLU(256, 128, kernel_size=1, padding=0)

        # Output 3×3 convolutions to smooth merged features
        self.out_n5 = ConvBNReLU(128, 128)
        self.out_n4 = ConvBNReLU(128, 128)
        self.out_n3 = ConvBNReLU(128, 128)

    def forward(self, p3, p4, p5):
        # Top-down pathway
        n5 = self.lat_p5(p5)                                            # /32
        n4 = self.lat_p4(p4) + F.interpolate(n5, scale_factor=2)       # /16
        n3 = self.lat_p3(p3) + F.interpolate(n4, scale_factor=2)       # /8

        return self.out_n3(n3), self.out_n4(n4), self.out_n5(n5)


# ---------------------------------------------------------------------------
# Detection Head
# ---------------------------------------------------------------------------
# Runs on each FPN level and predicts bounding boxes + objectness + classes.

class DetectionHead(nn.Module):
    """
    Shared-weight detection head applied independently to each FPN level.

    For every spatial location it predicts, per anchor:
        4  box offsets  (dx, dy, dw, dh)
        1  objectness score
        C  class scores
    → output channels per location = num_anchors × (5 + num_classes)
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_ch = num_anchors * (5 + num_classes)

        self.conv = nn.Sequential(
            ConvBNReLU(in_channels, 256),
            ConvBNReLU(256, 256),
        )
        self.pred = nn.Conv2d(256, out_ch, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.pred(self.conv(x))
        # Reshape → (B, num_anchors, H, W, 5 + num_classes)
        x = x.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x


# ---------------------------------------------------------------------------
# Post-Processing
# ---------------------------------------------------------------------------

def decode_predictions(raw, anchors, stride, conf_threshold=0.5):
    """
    Convert raw head output into absolute bounding boxes.

    Args:
        raw            : (B, A, H, W, 5+C)  — head output for one FPN level
        anchors        : list of (aw, ah) pairs for this level  (A entries)
        stride         : downsampling stride of this level (8, 16, or 32)
        conf_threshold : keep predictions above this objectness score

    Returns:
        List of dicts (one per image in the batch):
            {"boxes": (N,4) x1y1x2y2, "scores": (N,), "labels": (N,)}
    """
    B, A, H, W, _ = raw.shape
    device = raw.device

    # Build grid offsets
    gy, gx = torch.meshgrid(torch.arange(H, device=device),
                             torch.arange(W, device=device), indexing="ij")
    gx = gx.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    gy = gy.float().unsqueeze(0).unsqueeze(0)

    anchor_w = torch.tensor([a[0] for a in anchors], device=device).float()
    anchor_h = torch.tensor([a[1] for a in anchors], device=device).float()
    anchor_w = anchor_w.view(1, A, 1, 1)
    anchor_h = anchor_h.view(1, A, 1, 1)

    bx = (torch.sigmoid(raw[..., 0]) + gx) * stride
    by = (torch.sigmoid(raw[..., 1]) + gy) * stride
    bw = torch.exp(raw[..., 2]) * anchor_w
    bh = torch.exp(raw[..., 3]) * anchor_h

    obj  = torch.sigmoid(raw[..., 4])               # (B,A,H,W)
    cls  = torch.softmax(raw[..., 5:], dim=-1)       # (B,A,H,W,C)
    scores, labels = cls.max(dim=-1)                 # (B,A,H,W)
    scores = scores * obj

    # x1y1x2y2
    x1 = bx - bw / 2
    y1 = by - bh / 2
    x2 = bx + bw / 2
    y2 = by + bh / 2

    results = []
    for b in range(B):
        mask   = scores[b] > conf_threshold          # (A,H,W)
        boxes  = torch.stack([x1[b], y1[b], x2[b], y2[b]], dim=-1)[mask]
        sc     = scores[b][mask]
        lb     = labels[b][mask]
        results.append({"boxes": boxes, "scores": sc, "labels": lb})
    return results


def apply_nms(detections, iou_threshold=0.45):
    """
    Apply Non-Maximum Suppression per class to a list of detection dicts.

    Args:
        detections   : list of dicts with keys "boxes", "scores", "labels"
        iou_threshold: IoU threshold for NMS

    Returns:
        Same structure as input, with duplicates removed.
    """
    results = []
    for det in detections:
        boxes, scores, labels = det["boxes"], det["scores"], det["labels"]
        if boxes.numel() == 0:
            results.append(det)
            continue

        keep_idx = []
        for cls_id in labels.unique():
            mask     = labels == cls_id
            keep     = ops.nms(boxes[mask], scores[mask], iou_threshold)
            orig_idx = mask.nonzero(as_tuple=False).squeeze(1)
            keep_idx.append(orig_idx[keep])

        keep_idx = torch.cat(keep_idx)
        results.append({
            "boxes":  boxes[keep_idx],
            "scores": scores[keep_idx],
            "labels": labels[keep_idx],
        })
    return results


# ---------------------------------------------------------------------------
# Full Detector
# ---------------------------------------------------------------------------

# Default anchors (w, h) per FPN level, sized for a 416×416 input.
# These roughly correspond to small / medium / large objects.
DEFAULT_ANCHORS = {
    "small":  [(10, 13),  (16, 30),   (33, 23)],   # stride 8
    "medium": [(30, 61),  (62, 45),   (59, 119)],  # stride 16
    "large":  [(116, 90), (156, 198), (373, 326)],  # stride 32
}


class ImageDetector(nn.Module):
    """
    End-to-end single-shot object detector.

    Architecture:
        Input Image (3 × H × W)
             ↓
        [Backbone]  →  P3, P4, P5  (multi-scale feature maps)
             ↓
        [FPN Neck]  →  N3, N4, N5  (fused, equal-channel feature maps)
             ↓
        [Det. Head] →  raw predictions at each scale
             ↓
        [Post-Proc] →  decoded boxes + NMS  (inference only)

    Args:
        num_classes     : number of object categories
        num_anchors     : anchors per location (default 3)
        anchors         : dict with keys "small"/"medium"/"large",
                          each a list of (w, h) tuples
        conf_threshold  : objectness threshold for inference
        iou_threshold   : NMS IoU threshold for inference
    """

    def __init__(
        self,
        num_classes,
        num_anchors=3,
        anchors=None,
        conf_threshold=0.5,
        iou_threshold=0.45,
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.num_anchors     = num_anchors
        self.anchors         = anchors or DEFAULT_ANCHORS
        self.conf_threshold  = conf_threshold
        self.iou_threshold   = iou_threshold

        self.backbone = Backbone()
        self.neck     = FPN()

        # One shared-weight head per FPN level
        self.head_small  = DetectionHead(128, num_anchors, num_classes)
        self.head_medium = DetectionHead(128, num_anchors, num_classes)
        self.head_large  = DetectionHead(128, num_anchors, num_classes)

    def forward(self, x):
        """
        Args:
            x : (B, 3, H, W)  — normalised input image tensor

        Returns (training):
            Tuple of three raw prediction tensors:
                small  → (B, A, H/8,  W/8,  5+C)
                medium → (B, A, H/16, W/16, 5+C)
                large  → (B, A, H/32, W/32, 5+C)

        Returns (eval / inference):
            List[dict] — one dict per image, after NMS:
                {"boxes": (N,4), "scores": (N,), "labels": (N,)}
        """
        # Backbone
        p3, p4, p5 = self.backbone(x)

        # Neck
        n3, n4, n5 = self.neck(p3, p4, p5)

        # Heads
        raw_small  = self.head_small(n3)
        raw_medium = self.head_medium(n4)
        raw_large  = self.head_large(n5)

        if self.training:
            return raw_small, raw_medium, raw_large

        # ---- Inference path ------------------------------------------------
        # Decode predictions at each scale
        dets_s = decode_predictions(raw_small,  self.anchors["small"],  stride=8,  conf_threshold=self.conf_threshold)
        dets_m = decode_predictions(raw_medium, self.anchors["medium"], stride=16, conf_threshold=self.conf_threshold)
        dets_l = decode_predictions(raw_large,  self.anchors["large"],  stride=32, conf_threshold=self.conf_threshold)

        # Merge predictions across scales per image
        merged = []
        for s, m, l in zip(dets_s, dets_m, dets_l):
            boxes  = torch.cat([s["boxes"],  m["boxes"],  l["boxes"]],  dim=0)
            scores = torch.cat([s["scores"], m["scores"], l["scores"]], dim=0)
            labels = torch.cat([s["labels"], m["labels"], l["labels"]], dim=0)
            merged.append({"boxes": boxes, "scores": scores, "labels": labels})

        # Apply NMS
        return apply_nms(merged, iou_threshold=self.iou_threshold)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_CLASSES = 20  # e.g. PASCAL VOC
    BATCH_SIZE  = 2
    IMG_SIZE    = 416

    model = ImageDetector(num_classes=NUM_CLASSES)
    dummy = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)

    # --- Training mode ---
    model.train()
    raw_s, raw_m, raw_l = model(dummy)
    print("=== Training outputs ===")
    print(f"  small  head : {tuple(raw_s.shape)}")   # (2, 3, 52, 52, 25)
    print(f"  medium head : {tuple(raw_m.shape)}")   # (2, 3, 26, 26, 25)
    print(f"  large  head : {tuple(raw_l.shape)}")   # (2, 3, 13, 13, 25)

    # --- Inference mode ---
    model.eval()
    with torch.no_grad():
        detections = model(dummy)
    print("\n=== Inference outputs (after NMS) ===")
    for i, det in enumerate(detections):
        n = det["boxes"].shape[0]
        print(f"  image {i}: {n} detection(s)")
        if n:
            print(f"    boxes  : {det['boxes'].shape}")
            print(f"    scores : {det['scores'].shape}")
            print(f"    labels : {det['labels'].shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
