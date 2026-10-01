import os
import numpy as np
import torch
from torchvision.transforms import v2 as T
from engine_modi_ioutypes import train_one_epoch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import resnet50
import os.path as osp
import torch.nn as nn
import time
import math
from matplotlib.colors import hsv_to_rgb
from collections import OrderedDict
# from skimage.transform import resize

dropfea='loess_patch512_6-channel_dem_hsv_pretrained'
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

class SegmentationTransforms:
    def __init__(self, patch_size=512):
        self.patch_size = patch_size
        
        # Geometric transforms that should be applied to both image and mask
        self.geometric_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=30),
            T.RandomAffine(
                degrees=0,
                scale=(0.8, 1.2),
                translate=(0.1, 0.1)
            )])
        
        # Intensity transforms that should only be applied to the image
        # self.intensity_transforms = T.Compose([
        #     T.ColorJitter(
        #         brightness=0.2,
        #         contrast=0.2
        #     ),
        #     # T.GaussianNoise(std=0.01)
        # ])
    
    def __call__(self, image, mask):
        # Convert both to tensors first
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)
        
        # Stack image and mask along channel dimension to apply same geometric transforms
        stacked = torch.cat([image, mask], dim=0)
        stacked = self.geometric_transforms(stacked)
        
        # Split back into image and mask
        image, mask = torch.split(stacked, [image.shape[0], mask.shape[0]], dim=0)
        
        # Apply intensity transforms only to image
        # image = self.intensity_transforms(image)
        
        return image, mask

class LandslideDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images_6channel"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Images_6channel", self.imgs[idx])
        
        mask_path = os.path.join(self.root, "Masks", self.masks[idx])
        img = np.load(img_path)
        img = RGB_to_01(img)
        
        mask = np.load(mask_path)
        obj_ids, counts = np.unique(mask,return_counts=True)
        valid_indices=(obj_ids!=0) & (counts>10)
        obj_ids = obj_ids[valid_indices]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        valid_indices = []  # Track which objects have valid boxes
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # Ensure positive width and height
            width = xmax - xmin
            height = ymax - ymin
            
            if width <= 0 or height <= 0:
                print(f"Warning: Invalid box for object {i}: width={width}, height={height}")
                # Skip invalid boxes
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            valid_indices.append(i)
        num_objs = len(boxes)

        if num_objs == 0:
            print("Warning: No valid objects found in this image")
            return None
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Filter masks to only include valid objects
        valid_masks = [masks[i] for i in valid_indices]
        masks = torch.as_tensor(valid_masks, dtype=torch.uint8)

        image_id=osp.basename(img_path).split('.')[0]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

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
def get_model_instance_segmentation(num_classes, num_channels):
    model = MultiChannelMaskRCNN(num_classes, num_channels, pretrained_backbone=True)
    
    in_features = model.model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

    return model

def collate_fn(batch):
    # Remove None samples
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None  # optional: raise Exception("Empty batch")

    return tuple(zip(*batch))  # standard collate: returns (images, targets, image_paths)

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('The model will be running on', device, 'device')

num_classes = 2
num_channels = 6
transforms_t = T.Compose([T.ToTensor()])
transforms_t_train = SegmentationTransforms()

dataset = LandslideDataset('features_labels', transforms_t)

dataset_test = LandslideDataset('features_labels_test', transforms_t)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=1,
    collate_fn=collate_fn)
def adapt_first_conv_weights(optical_conv_weights, num_input_channels, target_shape):
    """
    Adapt the first convolutional layer weights from 3 channels to multi-channel input.
    
    Args:
        optical_conv_weights: Original conv1 weights (out_channels, 3, kernel_h, kernel_w)
        num_input_channels: Target number of input channels
        target_shape: Expected shape of the new conv layer
    
    Returns:
        Adapted weights for multi-channel input
    """
    out_channels, in_channels, kernel_h, kernel_w = optical_conv_weights.shape
    
    # Create new weight tensor with the target shape
    new_weights = torch.zeros(target_shape)
    
    if num_input_channels == 3:
        # If target is also 3 channels, use original weights
        new_weights = optical_conv_weights.clone()
    elif num_input_channels > 3:
        new_weights[:, :3, :, :] = optical_conv_weights
        # Replicate the RGB weights for additional channels
        for i in range(3, num_input_channels):
            channel_idx = i % 3  # Cycle through R, G, B
            new_weights[:, i, :, :] = optical_conv_weights[:, channel_idx, :, :]
    else:
        # If target has fewer channels, use subset of original weights
        new_weights = optical_conv_weights[:, :num_input_channels, :, :].clone()
    
    return new_weights

multi_model = get_model_instance_segmentation(num_classes, num_channels)
multi_state_dict = multi_model.state_dict()
# 1. Load the full optical model
optical_model = torch.load("landslides_maskrcnn_model.pth", map_location="cpu")
optical_state_dict = optical_model.state_dict()
# Create a new state dict for loading
new_state_dict = OrderedDict()

# Process each parameter
for key, value in optical_state_dict.items():
    key_multi='model.'+key
    if key_multi in multi_state_dict:
        # Handle the first conv layer specially (backbone.body.conv1.weight)
        if 'model.backbone.body.conv1.weight' in key_multi or 'model.model.backbone.body.conv1.weight' in key_multi:
            new_weight = adapt_first_conv_weights(
                value, num_channels, multi_state_dict[key_multi].shape
            )
            new_state_dict[key_multi] = new_weight
        else:
            # For all other layers, use weights directly if shapes match
            if value.shape == multi_state_dict[key_multi].shape:
                new_state_dict[key_multi] = value
            else:
                print(f"Shape mismatch for {key_multi}: optical {value.shape} vs multi {multi_state_dict[key_multi].shape}")
                # Keep the randomly initialized weights for mismatched layers
                new_state_dict[key_multi] = multi_state_dict[key_multi]
    else:
        print(f"Key {key_multi} not found in multi-modal model")

# Load the adapted weights
multi_model.load_state_dict(new_state_dict, strict=False)
print("Successfully loaded optical model weights into multi-modal model!")
model=multi_model
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, #0.005,
                            momentum=0.9, weight_decay=0.0005)

num_epochs = 200
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)

losses_classifier=[]
losses_box_reg=[]
losses_mask=[]
losses_objectness=[]
losses_rpn_box_reg=[]

startTime = time.time()
for epoch in range(num_epochs):
    
    loss_train=train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    train_dict=loss_train[1]
    for key, value in train_dict.items():
        train_dict[key] = value.detach().cpu().numpy()

    losses_classifier.append(train_dict['loss_classifier'])
    losses_box_reg.append(train_dict['loss_box_reg'])
    losses_mask.append(train_dict['loss_mask'])
    losses_objectness.append(train_dict['loss_objectness'])
    losses_rpn_box_reg.append(train_dict['loss_rpn_box_reg'])
    
    lr_scheduler.step()
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}hours".format(
	(endTime - startTime)/3600.0))
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) 
para_total=count_parameters(model)
exponent = int(math.floor(math.log10(para_total)))
coefficient = para_total / (10 ** exponent)
print(f"Total trainable parameters: {coefficient:.2f} × 10^{exponent}")

# save the model
torch.save(model, 'outputs/landslides_maskrcnn_model_'+dropfea+'.pth')