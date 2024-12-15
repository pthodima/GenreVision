import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

from torchvision.models import resnet

class GenreConvHead(nn.Module):
    def __init__(self, num_genres, backbone_out_feats_dims, pretrained=True):
        super(GenreConvHead, self).__init__()

        # Convolutional head
        self.conv1 = nn.Conv2d(backbone_out_feats_dims, 256, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)  # Second convolutional layer
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)   # Third convolutional layer

        # Global Average Pooling (GAP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Outputs a 1x1 feature map

        self.dropout = nn.Dropout(0.5)  # Dropout layer for regularization

        # Final fully connected layer for genre classification
        self.fc = nn.Linear(64, num_genres)  # Output layer for multi-label classification


    def forward(self, x):
        # Pass through the convolutional layers
        x = F.relu(self.conv1(x))  # ReLU activation after first convolutional layer
        x = F.relu(self.conv2(x))  # ReLU activation after second convolutional layer
        x = F.relu(self.conv3(x))  # ReLU activation after third convolutional layer

        # Apply Global Average Pooling (GAP) to reduce spatial dimensions
        x = self.global_avg_pool(x)  # Shape: [batch_size, 64, 1, 1]
        x = x.view(x.size(0), -1)    # Flatten the output to (batch_size, 64)

        # Apply dropout for regularization
        x = self.dropout(x)

        # Apply the final fully connected layer
        x = self.fc(x)

        return x

class MovieClassifier(nn.Module):
    def __init__(self, backbone, backbone_freeze_bn, backbone_out_feats_dims, num_genres, train_cfg, test_cfg, class_weights):
        super().__init__()
        assert backbone in ("resnet18", "resnet34")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.backbone = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        self.genre_head = GenreConvHead(num_genres, backbone_out_feats_dims)
        self.class_weights = class_weights


    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    def forward(self, images, targets):

        features = self.backbone(images)

        genre_logits = self.genre_head(features)

        if self.training:
            targets = targets.float()
            loss_criterion = BCEWithLogitsLoss(pos_weight=self.class_weights)
            genre_loss = loss_criterion(genre_logits, targets)
            return {"genre_loss": genre_loss, "final_loss": genre_loss}
        else:
           return genre_logits
