import os
import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import resnet50
import torch.nn as nn
# from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, Dice, BinaryPrecision, BinaryRecall
from matplotlib.colors import hsv_to_rgb

def normalize(array):
    min_val = np.nanmin(array)
    max_val = np.nanmax(array)
    if max_val == min_val:
        return np.zeros_like(array)
    else:
        normalized = (array - min_val) / (max_val - min_val)
        return normalized

def RGB_to_01(image_array):
    image_array = image_array.copy()  # avoid modifying original input
    image_array[:, :, :3] = image_array[:, :, :3] / 255.0
    H = normalize(image_array[:, :, 4])      # slope → hue
    S = normalize(image_array[:, :, 5])      # curvature → saturation
    V = normalize(image_array[:, :, 3])            # elevation → value
    # --- Stack HSV ---
    HSV = np.dstack([H, S, V])
    # --- Convert HSV → RGB ---
    RGB = hsv_to_rgb(HSV)
    image_array[:, :, 3:] = RGB
    return image_array

class MultiChannelMaskRCNN(nn.Module):
    """
    Mask R-CNN modified for multi-channel input
    """
    def __init__(self, num_classes, num_input_channels, pretrained_backbone=True):
        super(MultiChannelMaskRCNN, self).__init__()
        
        self.num_input_channels = num_input_channels
        
        if pretrained_backbone:
            # Load pretrained ResNet50 backbone
            backbone = resnet_fpn_backbone('resnet50', pretrained=True)
            
            # Modify the first conv layer to accept multi-channel input
            original_conv1 = backbone.body.conv1
            
            # Create new conv layer with desired input channels
            new_conv1 = nn.Conv2d(
                num_input_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )
            
            # Initialize weights for new conv layer
            if num_input_channels == 3:
                # If 3 channels, copy original weights
                new_conv1.weight.data = original_conv1.weight.data
                if original_conv1.bias is not None:
                    new_conv1.bias.data = original_conv1.bias.data
            else:
                # For multi-channel, initialize using different strategies
                self._initialize_first_conv_weights(new_conv1, original_conv1)
            
            # Replace the first conv layer
            backbone.body.conv1 = new_conv1
            
        else:
            # Create backbone from scratch
            backbone = self._create_custom_backbone(num_input_channels)
        
        # Create custom transform for multi-channel input
        custom_transform = self._create_custom_transform()
        self.backbone = backbone
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes)
        self.model.transform=custom_transform
    
    def _create_custom_transform(self, num_channels=6):
        """
        Create custom transform for multi-channel input
        """
        # For multi-channel, we need to define normalization statistics
        if num_channels == 3:
            # Use ImageNet statistics for RGB
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean_rgb = [0.485, 0.456, 0.406]  # ImageNet RGB means
            std_rgb = [0.229, 0.224, 0.225]   # ImageNet RGB stds
            mean = mean_rgb + mean_rgb
            std = std_rgb + std_rgb
        
        # Create custom transform
        transform = GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=mean,
            image_std=std,
        )
        
        return transform
    
    def _initialize_first_conv_weights(self, new_conv, original_conv):
        """
        Initialize weights for the new multi-channel conv layer
        """
        with torch.no_grad():
            if self.num_input_channels > 3:
                new_conv.weight.data[:, :3, :, :] = original_conv.weight.data
                new_conv.weight.data[:, 3:, :, :] = original_conv.weight.data
            else:
                # For fewer channels, use subset of original weights
                new_conv.weight.data = original_conv.weight.data[:, :self.num_input_channels, :, :]
            # Handle bias
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data
    
    def _create_custom_backbone(self, num_input_channels):
        """
        Create a custom backbone from scratch (without pretrained weights)
        """
        # Create ResNet50 backbone
        resnet = resnet50(pretrained=False)
        
        # Modify first conv layer
        resnet.conv1 = nn.Conv2d(
            num_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Create FPN backbone
        backbone = resnet_fpn_backbone('resnet50', pretrained=False)
        backbone.body.conv1 = resnet.conv1
        
        return backbone
    
    def forward(self, images, targets=None):
        """
        Forward pass
        Args:
            images: List of tensors or tensor of shape (N, C, H, W)
            targets: List of dicts for training (optional)
        """
        return self.model(images, targets)
dropfea='loess_patch512_6-channel_dem_hsv_pretrained'
model = torch.load('outputs/landslides_maskrcnn_model_'+dropfea+'.pth')
transforms_t = T.Compose([T.ToTensor()])
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#################### test set results ###########################################
threshold=np.arange(0,0.99,0.1)
iou=np.zeros(len(threshold))
f1_score_list=np.zeros(len(threshold))
dice_score_list=np.zeros(len(threshold))
precision_list=np.zeros(len(threshold))
recall_list=np.zeros(len(threshold))
valfolder='./features_labels_test'
imagePaths=os.listdir(os.path.join(valfolder, "Images_6channel"))
alpha = 0.3
beta = 1 - alpha
def compute_confusion(pred, gt):
    """
    pred, gt: torch.Tensor of shape (H, W)
              values in {0,1} or bool
    """
    pred = pred.bool()
    gt   = gt.bool()

    tp = (pred & gt).sum().item()
    fp = (pred & ~gt).sum().item()
    fn = (~pred & gt).sum().item()

    return tp, fp, fn


def precision_recall_f1(tp, fp, fn, eps=1e-8):
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou  = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return precision, recall, f1, iou, dice

with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    
    for i in range(len(threshold)):
        thresh=threshold[i]
        TP = FP = FN = 0
        for path in imagePaths:
            img_path = os.path.join(valfolder, "Images_6channel", path)
            # mask_name=path.replace('ge_image','fault_image')
            mask_path = os.path.join(valfolder, "Masks", path)
            origMask = np.load(mask_path)
            gt = (origMask > 0).astype(np.uint8)
            
            img = np.load(img_path)
            img = RGB_to_01(img)
            img = transforms_t(img)
            img = img.to(device)
            with torch.no_grad():
                pred = model([img])
                if len(pred[0]["boxes"]) == 0:
                    print("No predictions found")
                    pred_bin = np.zeros_like(gt)
                    
                else:
                    masks = pred[0]["masks"].squeeze(1).cpu().numpy()
                    pred_bin = np.zeros_like(gt)
                    for k in range(masks.shape[0]):
                        pred_bin[masks[k] > thresh] = 1
                pred_tensor = torch.from_numpy(pred_bin)
                gt_tensor   = torch.from_numpy(gt)
                TP += ((pred_tensor == 1) & (gt_tensor == 1)).sum().item()
                FP += ((pred_tensor == 1) & (gt_tensor == 0)).sum().item()
                FN += ((pred_tensor == 0) & (gt_tensor == 1)).sum().item()
        denominator = TP + alpha * FP + beta * FN
        precision_list[i] = TP / (TP + FP)
        recall_list[i]    = TP / (TP + FN)
        f1_score_list[i]  = 2 * TP / (2 * TP + FP + FN)
        iou[i]            = TP / (TP + FP + FN)
        dice_score_list[i]= TP / denominator

thresh_optimal=threshold[np.argmax(dice_score_list)]
print('iou_val='+str(np.amax(iou)))
print('Tversky-score_val='+str(np.amax(dice_score_list)))
print('Precision_val='+str(np.amax(precision_list)))
print('recall_val='+str(np.amax(recall_list)))