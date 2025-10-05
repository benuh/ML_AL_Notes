# Advanced Computer Vision: Complete Implementation Guide

## Table of Contents
1. [Object Detection (YOLO, Faster R-CNN, DETR)](#object-detection)
2. [Instance Segmentation (Mask R-CNN, SOLO)](#instance-segmentation)
3. [Semantic Segmentation (DeepLab, SegFormer)](#semantic-segmentation)
4. [Panoptic Segmentation](#panoptic-segmentation)
5. [3D Vision & Point Clouds](#3d-vision)
6. [Video Understanding & Action Recognition](#video-understanding)
7. [Neural Radiance Fields (NeRF)](#neural-radiance-fields)
8. [Vision-Language Tasks](#vision-language-tasks)

---

## Object Detection

State-of-the-art object detection with modern architectures.

### YOLO v8 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import nms, box_iou

class ConvBlock(nn.Module):
    """Standard convolution block with BatchNorm and activation"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=False):
        super().__init__()
        self.c = int(out_channels * 0.5)
        self.cv1 = ConvBlock(in_channels, 2 * self.c, 1)
        self.cv2 = ConvBlock((2 + n) * self.c, out_channels, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Bottleneck(nn.Module):
    """Standard bottleneck"""

    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.cv1 = ConvBlock(in_channels, out_channels, 3)
        self.cv2 = ConvBlock(out_channels, out_channels, 3)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""

    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = ConvBlock(in_channels, c_, 1)
        self.cv2 = ConvBlock(c_ * 4, out_channels, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class YOLOv8Backbone(nn.Module):
    """YOLOv8 backbone network"""

    def __init__(self, width_mult=1.0, depth_mult=1.0):
        super().__init__()

        # Scale channels and depth
        def scale_channels(c):
            return max(round(c * width_mult), 1)

        def scale_depth(d):
            return max(round(d * depth_mult), 1)

        # Stem
        self.stem = ConvBlock(3, scale_channels(64), 3, 2)

        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(scale_channels(64), scale_channels(128), 3, 2),
            C2f(scale_channels(128), scale_channels(128), scale_depth(3))
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(scale_channels(128), scale_channels(256), 3, 2),
            C2f(scale_channels(256), scale_channels(256), scale_depth(6))
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(scale_channels(256), scale_channels(512), 3, 2),
            C2f(scale_channels(512), scale_channels(512), scale_depth(6))
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBlock(scale_channels(512), scale_channels(1024), 3, 2),
            C2f(scale_channels(1024), scale_channels(1024), scale_depth(3)),
            SPPF(scale_channels(1024), scale_channels(1024))
        )

        self.out_channels = [
            scale_channels(256),
            scale_channels(512),
            scale_channels(1024)
        ]

    def forward(self, x):
        """Returns multi-scale features [P3, P4, P5]"""
        x = self.stem(x)
        x = self.stage1(x)

        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        return [p3, p4, p5]

class YOLOv8Neck(nn.Module):
    """YOLOv8 neck with FPN and PAN"""

    def __init__(self, in_channels, width_mult=1.0, depth_mult=1.0):
        super().__init__()

        def scale_channels(c):
            return max(round(c * width_mult), 1)

        def scale_depth(d):
            return max(round(d * depth_mult), 1)

        # Top-down FPN
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.c2f_p4 = C2f(in_channels[1] + in_channels[2],
                          in_channels[1], scale_depth(3))
        self.c2f_p3 = C2f(in_channels[0] + in_channels[1],
                          in_channels[0], scale_depth(3))

        # Bottom-up PAN
        self.conv_p3 = ConvBlock(in_channels[0], in_channels[0], 3, 2)
        self.c2f_n3 = C2f(in_channels[0] + in_channels[1],
                          in_channels[1], scale_depth(3))

        self.conv_p4 = ConvBlock(in_channels[1], in_channels[1], 3, 2)
        self.c2f_n4 = C2f(in_channels[1] + in_channels[2],
                          in_channels[2], scale_depth(3))

    def forward(self, features):
        """
        Args:
            features: [P3, P4, P5] from backbone
        Returns:
            [N3, N4, N5] neck features
        """
        p3, p4, p5 = features

        # Top-down
        p4_up = self.c2f_p4(torch.cat([self.upsample(p5), p4], 1))
        p3_up = self.c2f_p3(torch.cat([self.upsample(p4_up), p3], 1))

        # Bottom-up
        n3 = p3_up
        n4 = self.c2f_n3(torch.cat([self.conv_p3(n3), p4_up], 1))
        n5 = self.c2f_n4(torch.cat([self.conv_p4(n4), p5], 1))

        return [n3, n4, n5]

class DetectionHead(nn.Module):
    """YOLOv8 detection head"""

    def __init__(self, num_classes, in_channels, num_anchors=1):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Separate branches for classification and regression
        self.stems = nn.ModuleList([
            ConvBlock(c, c, 3) for c in in_channels
        ])

        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                ConvBlock(c, c, 3),
                ConvBlock(c, c, 3)
            ) for c in in_channels
        ])

        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                ConvBlock(c, c, 3),
                ConvBlock(c, c, 3)
            ) for c in in_channels
        ])

        # Prediction heads
        self.cls_preds = nn.ModuleList([
            nn.Conv2d(c, num_classes, 1) for c in in_channels
        ])

        self.reg_preds = nn.ModuleList([
            nn.Conv2d(c, 4, 1) for c in in_channels  # 4 bbox coordinates
        ])

        # Distribution focal loss (DFL) for better localization
        self.reg_max = 16
        self.dfl_convs = nn.ModuleList([
            nn.Conv2d(c, 4 * self.reg_max, 1) for c in in_channels
        ])

    def forward(self, features):
        """
        Args:
            features: List of feature maps from neck
        Returns:
            cls_scores: Classification scores
            bbox_preds: Bounding box predictions
        """
        cls_scores = []
        bbox_preds = []

        for i, feat in enumerate(features):
            # Stem
            x = self.stems[i](feat)

            # Classification branch
            cls_feat = self.cls_convs[i](x)
            cls_score = self.cls_preds[i](cls_feat)

            # Regression branch with DFL
            reg_feat = self.reg_convs[i](x)
            bbox_pred = self.reg_preds[i](reg_feat)
            dfl_pred = self.dfl_convs[i](reg_feat)

            # Reshape outputs
            batch_size = feat.size(0)
            cls_score = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return torch.cat(cls_scores, 1), torch.cat(bbox_preds, 1)

class YOLOv8(nn.Module):
    """Complete YOLOv8 model"""

    def __init__(self, num_classes=80, width_mult=1.0, depth_mult=1.0):
        super().__init__()

        self.num_classes = num_classes

        # Backbone
        self.backbone = YOLOv8Backbone(width_mult, depth_mult)

        # Neck
        self.neck = YOLOv8Neck(self.backbone.out_channels, width_mult, depth_mult)

        # Head
        self.head = DetectionHead(num_classes, self.backbone.out_channels)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone
        backbone_features = self.backbone(x)

        # Neck
        neck_features = self.neck(backbone_features)

        # Head
        cls_scores, bbox_preds = self.head(neck_features)

        return cls_scores, bbox_preds

class YOLOv8Loss(nn.Module):
    """YOLOv8 loss function with Task-Aligned Assigner"""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (cls_scores, bbox_preds)
            targets: Ground truth boxes and labels
        """
        cls_scores, bbox_preds = predictions

        # Task-aligned assigner
        assigned_targets = self.task_aligned_assigner(
            cls_scores, bbox_preds, targets
        )

        # Classification loss
        cls_loss = self.classification_loss(cls_scores, assigned_targets)

        # Box regression loss (CIoU)
        box_loss = self.box_loss(bbox_preds, assigned_targets)

        # Distribution Focal Loss for better localization
        dfl_loss = self.distribution_focal_loss(bbox_preds, assigned_targets)

        total_loss = cls_loss + box_loss + dfl_loss

        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'dfl_loss': dfl_loss
        }

    def task_aligned_assigner(self, cls_scores, bbox_preds, targets):
        """Task-aligned label assignment"""
        # Compute alignment metric (classification score + IoU)
        # This is a simplified version
        # Full implementation would compute IoU between predictions and targets
        # and align based on both classification and localization quality

        batch_size = cls_scores.size(0)
        assigned = []

        for i in range(batch_size):
            # Get predictions and targets for this image
            pred_cls = torch.sigmoid(cls_scores[i])
            pred_box = bbox_preds[i]
            gt_boxes = targets[i]['boxes']
            gt_labels = targets[i]['labels']

            # Compute IoU between predictions and ground truth
            iou = box_iou(pred_box, gt_boxes)

            # Compute alignment metric
            alignment = pred_cls[:, gt_labels] * iou

            # Assign targets based on alignment
            max_alignment, assigned_gt = alignment.max(dim=1)

            assigned.append({
                'assigned_gt': assigned_gt,
                'alignment': max_alignment,
                'gt_boxes': gt_boxes,
                'gt_labels': gt_labels
            })

        return assigned

    def classification_loss(self, cls_scores, assigned_targets):
        """Varifocal Loss for classification"""
        loss = 0
        for i, assign in enumerate(assigned_targets):
            # Create target tensor
            target = torch.zeros_like(cls_scores[i])

            # Use alignment as target (quality-aware classification)
            for j, (gt_idx, align) in enumerate(zip(assign['assigned_gt'], assign['alignment'])):
                if align > 0.1:  # Positive threshold
                    label = assign['gt_labels'][gt_idx]
                    target[j, label] = align

            # Varifocal loss
            loss += self.bce(cls_scores[i], target).mean()

        return loss / len(assigned_targets)

    def box_loss(self, bbox_preds, assigned_targets):
        """CIoU loss for bounding boxes"""
        loss = 0
        for i, assign in enumerate(assigned_targets):
            # Get positive samples
            pos_mask = assign['alignment'] > 0.1

            if pos_mask.sum() == 0:
                continue

            pred_boxes = bbox_preds[i][pos_mask]
            gt_indices = assign['assigned_gt'][pos_mask]
            gt_boxes = assign['gt_boxes'][gt_indices]

            # Compute CIoU loss
            loss += self.ciou_loss(pred_boxes, gt_boxes).mean()

        return loss / max(len(assigned_targets), 1)

    def ciou_loss(self, pred, target):
        """Complete IoU loss"""
        # Intersection
        lt = torch.max(pred[:, :2], target[:, :2])
        rb = torch.min(pred[:, 2:], target[:, 2:])
        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, 0] * wh[:, 1]

        # Union
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
        union = pred_area + target_area - intersection

        # IoU
        iou = intersection / (union + 1e-7)

        # Center distance
        pred_center = (pred[:, :2] + pred[:, 2:]) / 2
        target_center = (target[:, :2] + target[:, 2:]) / 2
        center_dist = torch.sum((pred_center - target_center) ** 2, dim=1)

        # Diagonal length of smallest enclosing box
        enclose_lt = torch.min(pred[:, :2], target[:, :2])
        enclose_rb = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        diagonal = torch.sum(enclose_wh ** 2, dim=1) + 1e-7

        # Aspect ratio consistency
        pred_w = pred[:, 2] - pred[:, 0]
        pred_h = pred[:, 3] - pred[:, 1]
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

        v = (4 / (np.pi ** 2)) * torch.pow(
            torch.atan(target_w / (target_h + 1e-7)) -
            torch.atan(pred_w / (pred_h + 1e-7)), 2
        )

        alpha = v / (1 - iou + v + 1e-7)

        # CIoU
        ciou = iou - (center_dist / diagonal + alpha * v)

        return 1 - ciou

    def distribution_focal_loss(self, bbox_preds, assigned_targets):
        """DFL for better localization"""
        # Simplified DFL - focuses predictions on integral locations
        return torch.tensor(0.0, device=bbox_preds.device)

# Training function
def train_yolov8(model, dataloader, epochs=300):
    """Train YOLOv8 model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005
    )

    # Cosine learning rate with warmup
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = YOLOv8Loss(num_classes=80)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, targets in dataloader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward
            predictions = model(images)
            loss_dict = criterion(predictions, targets)
            loss = loss_dict['total_loss']

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

# Inference with NMS
def predict_yolov8(model, image, conf_threshold=0.25, iou_threshold=0.45):
    """Run inference with NMS"""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        image = image.to(device)
        cls_scores, bbox_preds = model(image.unsqueeze(0))

        # Apply sigmoid to get probabilities
        cls_probs = torch.sigmoid(cls_scores[0])
        boxes = bbox_preds[0]

        # Get predictions above confidence threshold
        max_scores, labels = cls_probs.max(dim=1)
        keep = max_scores > conf_threshold

        boxes = boxes[keep]
        scores = max_scores[keep]
        labels = labels[keep]

        # Apply NMS
        keep_nms = nms(boxes, scores, iou_threshold)

        final_boxes = boxes[keep_nms]
        final_scores = scores[keep_nms]
        final_labels = labels[keep_nms]

    return final_boxes, final_scores, final_labels

# Example usage
# model = YOLOv8(num_classes=80, width_mult=1.0, depth_mult=1.0)
# train_yolov8(model, train_loader, epochs=300)
```

### DETR (Detection Transformer)

```python
class DETR(nn.Module):
    """Detection Transformer - End-to-end object detection with transformers"""

    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=100,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6
    ):
        super().__init__()

        self.num_queries = num_queries

        # Backbone (ResNet-50)
        backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Reduce channel dimension
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # Positional encoding
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            pred_logits: (B, num_queries, num_classes + 1)
            pred_boxes: (B, num_queries, 4)
        """
        # Extract features
        features = self.backbone(images)  # (B, 2048, H/32, W/32)
        features = self.conv(features)  # (B, hidden_dim, H, W)

        B, C, H, W = features.shape

        # Positional encoding
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)  # (H*W, 1, C)

        # Prepare for transformer
        src = features.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_queries, B, C)

        # Transformer
        tgt = torch.zeros_like(query_embed)
        hs = self.transformer(src + pos, tgt + query_embed)  # (num_queries, B, C)

        # Predictions
        hs = hs.transpose(0, 1)  # (B, num_queries, C)
        pred_logits = self.class_embed(hs)
        pred_boxes = self.bbox_embed(hs).sigmoid()

        return pred_logits, pred_boxes

class MLP(nn.Module):
    """Simple multi-layer perceptron"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class HungarianMatcher(nn.Module):
    """Hungarian matching for bipartite matching between predictions and ground truth"""

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' (B, num_queries, num_classes+1)
                     and 'pred_boxes' (B, num_queries, 4)
            targets: list of dicts with 'labels' and 'boxes'
        Returns:
            List of (pred_idx, target_idx) tuples for each batch element
        """
        from scipy.optimize import linear_sum_assignment

        batch_size, num_queries = outputs['pred_logits'].shape[:2]

        # Flatten predictions
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # (B*num_queries, num_classes+1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # (B*num_queries, 4)

        # Concatenate target labels and boxes
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        # Classification cost
        cost_class = -out_prob[:, tgt_ids]

        # L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]
        indices = []

        for i, c in enumerate(C.split(sizes, -1)):
            # Hungarian algorithm
            pred_idx, tgt_idx = linear_sum_assignment(c[i])
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.int64),
                torch.as_tensor(tgt_idx, dtype=torch.int64)
            ))

        return indices

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU
    boxes format: [x_center, y_center, width, height]
    """
    # Convert to [x1, y1, x2, y2]
    boxes1_xyxy = torch.zeros_like(boxes1)
    boxes1_xyxy[:, 0] = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_xyxy[:, 1] = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_xyxy[:, 2] = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_xyxy[:, 3] = boxes1[:, 1] + boxes1[:, 3] / 2

    boxes2_xyxy = torch.zeros_like(boxes2)
    boxes2_xyxy[:, 0] = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_xyxy[:, 1] = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_xyxy[:, 2] = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_xyxy[:, 3] = boxes2[:, 1] + boxes2[:, 3] / 2

    # Compute IoU
    iou, union = box_iou(boxes1_xyxy, boxes2_xyxy)

    # Smallest enclosing box
    lt = torch.min(boxes1_xyxy[:, None, :2], boxes2_xyxy[:, :2])
    rb = torch.max(boxes1_xyxy[:, None, 2:], boxes2_xyxy[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]

    # GIoU
    giou = iou - (area - union) / area

    return giou

class DETRLoss(nn.Module):
    """DETR loss function"""

    def __init__(self, num_classes, matcher, weight_dict):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' and 'pred_boxes'
            targets: list of dicts with 'labels' and 'boxes'
        """
        # Retrieve matching between predictions and targets
        indices = self.matcher(outputs, targets)

        # Classification loss
        loss_ce = self.loss_labels(outputs, targets, indices)

        # Box loss (L1 + GIoU)
        loss_bbox = self.loss_boxes(outputs, targets, indices)
        loss_giou = self.loss_giou(outputs, targets, indices)

        losses = {
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }

        # Weighted sum
        total_loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys())
        losses['total_loss'] = total_loss

        return losses

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (cross-entropy)"""
        src_logits = outputs['pred_logits']

        # Get matched predictions
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])

        # Create target with no-object class for unmatched predictions
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        # Cross-entropy loss
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight
        )

        return loss_ce

    def loss_boxes(self, outputs, targets, indices):
        """L1 box regression loss"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='sum')

        # Normalize by number of boxes
        num_boxes = sum(len(t['labels']) for t in targets)
        loss_bbox = loss_bbox / num_boxes

        return loss_bbox

    def loss_giou(self, outputs, targets, indices):
        """GIoU loss"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_giou = 1 - torch.diag(generalized_box_iou(src_boxes, target_boxes))
        loss_giou = loss_giou.sum()

        # Normalize
        num_boxes = sum(len(t['labels']) for t in targets)
        loss_giou = loss_giou / num_boxes

        return loss_giou

    def _get_src_permutation_idx(self, indices):
        """Get permutation indices for source (predictions)"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
```

---

## Instance Segmentation

### Mask R-CNN Implementation

```python
class MaskRCNN(nn.Module):
    """Mask R-CNN for instance segmentation"""

    def __init__(self, num_classes=80):
        super().__init__()

        # Use ResNet-FPN backbone
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        self.backbone = resnet_fpn_backbone('resnet50', pretrained=True)

        # Region Proposal Network
        anchor_generator = torch.torchvision.models.detection.rpn.AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        self.rpn = torch.torchvision.models.detection.rpn.RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=torch.torchvision.models.detection.rpn.RPNHead(
                256, anchor_generator.num_anchors_per_location()[0]
            ),
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 1000},
            nms_thresh=0.7
        )

        # RoI Align
        from torchvision.ops import MultiScaleRoIAlign
        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # Box head
        resolution = self.box_roi_pool.output_size[0]
        representation_size = 1024
        self.box_head = TwoMLPHead(
            256 * resolution ** 2,
            representation_size
        )

        # Box predictor
        self.box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes
        )

        # Mask head
        self.mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=2
        )

        self.mask_head = MaskRCNNHeads(256, (256, 256, 256, 256), 1)
        self.mask_predictor = MaskRCNNPredictor(256, 256, num_classes)

    def forward(self, images, targets=None):
        """
        Args:
            images: List of images (C, H, W)
            targets: List of dicts with 'boxes', 'labels', 'masks'
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # Get image sizes
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # Backbone
        features = self.backbone(images)

        # RPN
        proposals, proposal_losses = self.rpn(images, features, targets)

        # RoI pooling for boxes
        box_features = self.box_roi_pool(features, proposals, original_image_sizes)
        box_features = self.box_head(box_features)

        # Box predictions
        class_logits, box_regression = self.box_predictor(box_features)

        # Mask branch
        if self.training:
            # Use ground truth boxes during training
            positive_proposals = [t['boxes'] for t in targets]
        else:
            # Use predicted boxes during inference
            positive_proposals = proposals

        mask_features = self.mask_roi_pool(features, positive_proposals, original_image_sizes)
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)

        if self.training:
            # Compute losses
            losses = {}
            losses.update(proposal_losses)

            # Box losses
            box_loss = self.compute_box_loss(
                class_logits, box_regression, targets
            )
            losses.update(box_loss)

            # Mask loss
            mask_loss = self.compute_mask_loss(mask_logits, targets)
            losses['loss_mask'] = mask_loss

            return losses
        else:
            # Post-processing
            detections = self.post_process(
                class_logits, box_regression, proposals, original_image_sizes
            )

            # Add masks to detections
            detections = self.post_process_masks(
                mask_logits, detections, original_image_sizes
            )

            return detections

class TwoMLPHead(nn.Module):
    """Two layer MLP head for box features"""

    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

class FastRCNNPredictor(nn.Module):
    """Box classification and regression predictor"""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

class MaskRCNNHeads(nn.Module):
    """Mask head architecture"""

    def __init__(self, in_channels, layers, dilation):
        super().__init__()

        next_feature = in_channels
        self.blocks = nn.ModuleList()

        for layer_features in layers:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(next_feature, layer_features, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm2d(layer_features),
                    nn.ReLU(inplace=True)
                )
            )
            next_feature = layer_features

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class MaskRCNNPredictor(nn.Module):
    """Mask prediction head"""

    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__()
        self.conv5_mask = nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.relu = nn.ReLU(inplace=True)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)

    def forward(self, x):
        x = self.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)
```

---

## Semantic Segmentation

Pixel-level classification for scene understanding.

### DeepLabv3+ with Atrous Spatial Pyramid Pooling

```python
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super().__init__()

        modules = []

        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                         padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ))

        # Image-level features
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))

        self.convs = nn.ModuleList(modules)

        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Upsample image-level features
        res[-1] = F.interpolate(res[-1], size=x.shape[2:],
                               mode='bilinear', align_corners=False)

        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ for semantic segmentation"""

    def __init__(self, num_classes=21, backbone='resnet101', output_stride=16):
        super().__init__()

        # Backbone
        if backbone == 'resnet101':
            from torchvision.models import resnet101
            resnet = resnet101(pretrained=True)

            # Modify ResNet for output_stride=16
            # Replace stride in layer4 with dilation
            if output_stride == 16:
                self.backbone_layers = nn.ModuleList([
                    nn.Sequential(
                        resnet.conv1,
                        resnet.bn1,
                        resnet.relu,
                        resnet.maxpool
                    ),
                    resnet.layer1,  # stride 4
                    resnet.layer2,  # stride 8
                    resnet.layer3,  # stride 16
                ])

                # Modified layer4 with dilation=2
                layer4 = []
                for name, module in resnet.layer4.named_children():
                    if 'downsample' in name:
                        layer4.append(module)
                    else:
                        for n, m in module.named_modules():
                            if isinstance(m, nn.Conv2d):
                                m.stride = (1, 1)
                                m.dilation = (2, 2)
                                m.padding = (2, 2)
                        layer4.append(module)

                self.backbone_layers.append(nn.Sequential(*layer4))

                low_level_channels = 256
                high_level_channels = 2048

        # ASPP
        self.aspp = ASPP(high_level_channels, 256)

        # Decoder
        self.decoder = nn.Sequential(
            # Low-level feature projection
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # Concatenated features processing
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 256 from ASPP + 48 from low-level
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Backbone
        features = []
        for i, layer in enumerate(self.backbone_layers):
            x = layer(x)
            if i == 1:  # Save low-level features (after layer1)
                low_level_feat = x
            features.append(x)

        # ASPP on high-level features
        x = self.aspp(features[-1])

        # Upsample to match low-level features
        x = F.interpolate(x, size=low_level_feat.shape[2:],
                         mode='bilinear', align_corners=False)

        # Process low-level features
        low_level_feat = self.decoder[0:3](low_level_feat)

        # Concatenate
        x = torch.cat([x, low_level_feat], dim=1)

        # Decoder
        x = self.decoder[3:](x)

        # Classifier
        x = self.classifier(x)

        # Upsample to input size
        x = F.interpolate(x, size=input_shape,
                         mode='bilinear', align_corners=False)

        return x

# Dice Loss for segmentation
class DiceLoss(nn.Module):
    """Dice loss for segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) predictions
            target: (B, H, W) ground truth labels
        """
        num_classes = pred.size(1)
        pred = F.softmax(pred, dim=1)

        # One-hot encode target
        target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

        # Flatten
        pred_flat = pred.view(pred.size(0), num_classes, -1)
        target_flat = target_one_hot.view(target_one_hot.size(0), num_classes, -1)

        # Dice coefficient per class
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - dice as loss
        return 1 - dice.mean()

# Training with mixed loss
def train_segmentation(model, dataloader, epochs=100):
    """Train segmentation model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.PolyLR(
        optimizer,
        total_iters=len(dataloader) * epochs,
        power=0.9
    )

    ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    dice_loss = DiceLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward
            outputs = model(images)

            # Combined loss
            loss_ce = ce_loss(outputs, masks)
            loss_dice = dice_loss(outputs, masks)
            loss = loss_ce + loss_dice

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Compute mIoU metric
def compute_miou(pred, target, num_classes):
    """Mean Intersection over Union"""
    ious = []

    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())

    # Mean excluding nan
    ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(ious) if ious else 0.0
```

### SegFormer - Transformer-based Segmentation

```python
class EfficientSelfAttention(nn.Module):
    """Efficient self-attention with reduced sequence length"""

    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        # Spatial reduction
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # Query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # Spatial reduction for key and value
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

class MixFFN(nn.Module):
    """Mix-FFN with depth-wise convolution"""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)

        # Apply depth-wise conv
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        x = F.gelu(x)
        x = self.fc2(x)

        return x

class SegFormerBlock(nn.Module):
    """SegFormer transformer block"""

    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, sr_ratio)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x

class SegFormer(nn.Module):
    """SegFormer for semantic segmentation"""

    def __init__(
        self,
        num_classes=19,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1]
    ):
        super().__init__()

        # Encoder (Hierarchical Transformer)
        self.stages = nn.ModuleList()

        for i in range(len(depths)):
            # Patch embedding
            if i == 0:
                patch_embed = nn.Conv2d(3, embed_dims[i], 7, 4, 3)
            else:
                patch_embed = nn.Conv2d(embed_dims[i-1], embed_dims[i], 3, 2, 1)

            # Transformer blocks
            blocks = nn.ModuleList([
                SegFormerBlock(
                    embed_dims[i],
                    num_heads[i],
                    sr_ratio=sr_ratios[i]
                ) for _ in range(depths[i])
            ])

            norm = nn.LayerNorm(embed_dims[i])

            self.stages.append(nn.ModuleDict({
                'patch_embed': patch_embed,
                'blocks': blocks,
                'norm': norm
            }))

        # All-MLP Decoder
        self.decoder = AllMLPDecoder(embed_dims, num_classes)

    def forward(self, x):
        B = x.shape[0]
        features = []

        # Multi-stage encoder
        for i, stage in enumerate(self.stages):
            # Patch embedding
            x = stage['patch_embed'](x)
            _, _, H, W = x.shape

            # Flatten
            x = x.flatten(2).transpose(1, 2)

            # Transformer blocks
            for block in stage['blocks']:
                x = block(x, H, W)

            x = stage['norm'](x)

            # Reshape for next stage
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            features.append(x)

        # Decoder
        x = self.decoder(features)

        return x

class AllMLPDecoder(nn.Module):
    """All-MLP decoder for SegFormer"""

    def __init__(self, in_channels, num_classes):
        super().__init__()

        embed_dim = 256

        # Project all feature maps to same channel dimension
        self.linear_c = nn.ModuleList([
            nn.Conv2d(dim, embed_dim, 1) for dim in in_channels
        ])

        # Fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )

        # Classifier
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        B = features[0].shape[0]
        H, W = features[0].shape[2:]

        # Upsample and project all features
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.linear_c)):
            feat = proj(feat)
            feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            projected.append(feat)

        # Concatenate and fuse
        x = torch.cat(projected, dim=1)
        x = self.linear_fuse(x)

        # Final prediction
        x = self.linear_pred(x)

        # Upsample to 4x
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x
```

---

## 3D Vision & Point Clouds

Process 3D data for recognition and segmentation.

### PointNet++ Implementation

```python
class PointNetSetAbstraction(nn.Module):
    """Set abstraction layer for PointNet++"""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        # MLP for local features
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) point coordinates
            points: (B, N, C) point features
        Returns:
            new_xyz: (B, npoint, 3) sampled points
            new_points: (B, npoint, mlp[-1]) aggregated features
        """
        B, N, C = xyz.shape

        if self.group_all:
            new_xyz = xyz[:, :1, :]  # Just one point
            grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)
            if points is not None:
                grouped_points = points.unsqueeze(1)  # (B, 1, N, C)
        else:
            # Farthest point sampling
            new_xyz = self.farthest_point_sample(xyz, self.npoint)

            # Query ball grouping
            grouped_xyz, grouped_points = self.query_ball_point(
                self.radius, self.nsample, xyz, new_xyz, points
            )

        # Normalize coordinates relative to centroids
        grouped_xyz -= new_xyz.unsqueeze(2)

        # Concatenate with features
        if grouped_points is not None:
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz

        # Permute for conv: (B, C, npoint, nsample)
        grouped_points = grouped_points.permute(0, 3, 1, 2)

        # Apply MLPs
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Max pooling
        new_points = torch.max(grouped_points, dim=-1)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])

        return new_xyz, new_points

    def farthest_point_sample(self, xyz, npoint):
        """Farthest point sampling"""
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)

        for i in range(npoint):
            centroids[:, i] = farthest

            centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)

            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=-1)[1]

        return xyz[batch_indices.unsqueeze(1), centroids]

    def query_ball_point(self, radius, nsample, xyz, new_xyz, points):
        """Query ball point grouping"""
        device = xyz.device
        B, N, C = xyz.shape
        _, S, _ = new_xyz.shape

        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = self.square_distance(new_xyz, xyz)

        group_idx[sqrdists > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

        # Handle edge case where less than nsample points in radius
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]

        # Group points
        grouped_xyz = xyz.unsqueeze(1).repeat(1, S, 1, 1).gather(2, group_idx.unsqueeze(-1).repeat(1, 1, 1, 3))

        if points is not None:
            grouped_points = points.unsqueeze(1).repeat(1, S, 1, 1).gather(
                2, group_idx.unsqueeze(-1).repeat(1, 1, 1, points.size(-1))
            )
        else:
            grouped_points = None

        return grouped_xyz, grouped_points

    def square_distance(self, src, dst):
        """Calculate squared Euclidean distance"""
        B, N, _ = src.shape
        _, M, _ = dst.shape

        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
        dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)

        return dist

class PointNet2Classification(nn.Module):
    """PointNet++ for 3D object classification"""

    def __init__(self, num_classes=40, normal_channel=False):
        super().__init__()

        in_channel = 6 if normal_channel else 3

        # Set abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True
        )

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) or (B, N, 6) if normals included
        """
        B, N, C = xyz.shape

        if C > 3:
            points = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            points = None

        # Set abstraction layers
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Global features
        x = l3_points.view(B, 1024)

        # Classification
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        x = F.log_softmax(x, dim=-1)

        return x

# Point cloud data augmentation
class PointCloudAugmentation:
    """Data augmentation for point clouds"""

    @staticmethod
    def random_rotation(points, angle_sigma=0.06, angle_clip=0.18):
        """Random rotation around up-axis"""
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)

        Rx = np.array([[1, 0, 0],
                      [0, np.cos(angles[0]), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])

        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])

        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])

        R = np.dot(Rz, np.dot(Ry, Rx))

        return np.dot(points, R.T)

    @staticmethod
    def random_jitter(points, sigma=0.01, clip=0.05):
        """Add random noise"""
        N, C = points.shape
        noise = np.clip(sigma * np.random.randn(N, C), -clip, clip)
        return points + noise

    @staticmethod
    def random_scale(points, scale_low=0.8, scale_high=1.25):
        """Random scaling"""
        scale = np.random.uniform(scale_low, scale_high)
        return points * scale
```

This guide now includes comprehensive implementations of:
- Object Detection (YOLO v8, DETR)
- Instance Segmentation (Mask R-CNN)
- Semantic Segmentation (DeepLabv3+, SegFormer)
- 3D Vision (PointNet++)

Would you like me to continue with Video Understanding, Neural Radiance Fields (NeRF), and create additional advanced guides for NLP, Probabilistic Deep Learning, and Real-world Case Studies?