"""One photo classifier and a multi-lesion detector sharing ResNet-18/FPN."""
import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class FPNPhotoEncoder(nn.Module):
    output_dim=256

    def __init__(self,min_size=512,max_size=1024):
        super().__init__()
        self.backbone=resnet_fpn_backbone(backbone_name='resnet18',weights=None,trainable_layers=5)
        self.transform=GeneralizedRCNNTransform(min_size,max_size,[.485,.456,.406],[.229,.224,.225])
        self.canvas=((max_size+31)//32)*32

    def forward(self,images):
        transformed,_=self.transform(list(images),None)
        h,w=transformed.tensors.shape[-2:]
        tensor=torch.nn.functional.pad(transformed.tensors,(0,self.canvas-w,0,self.canvas-h))
        return self.backbone(tensor)['3'].mean(dim=(-2,-1))


class LesionDetector(nn.Module):
    feature_dim = 256

    def __init__(self, config, pretrained=False, task="joint"):
        super().__init__()
        self.task, self.config = task, config
        backbone = resnet_fpn_backbone(backbone_name="resnet18", weights=
            ResNet18_Weights.IMAGENET1K_V1 if pretrained else None, trainable_layers=5)
        self.detector = FasterRCNN(backbone, num_classes=2,
            min_size=config["min_size"], max_size=config["max_size"],
            box_score_thresh=config["box_score_thresh"], box_nms_thresh=config["box_nms_thresh"],
            box_detections_per_img=config["detections_per_image"])
        self.classifier = nn.Linear(256, 4)

    def encoded(self, images, targets=None):
        shapes = [tuple(image.shape[-2:]) for image in images]
        transformed, targets = self.detector.transform(images, targets)
        # A fixed padded canvas makes the global classification feature independent
        # of the aspect ratios of other photos in the same batch.
        canvas=((self.config['max_size']+31)//32)*32
        height,width=transformed.tensors.shape[-2:]
        if height>canvas or width>canvas:
            raise ValueError('Detection transform exceeded fixed canvas')
        transformed.tensors=torch.nn.functional.pad(transformed.tensors,(0,canvas-width,0,canvas-height))
        features = self.detector.backbone(transformed.tensors)
        pooled = features["3"].mean(dim=(-2, -1))
        return pooled, transformed, features, targets, shapes

    def image_features(self, images):
        return self.encoded(images)[0]

    def forward(self, images, targets=None):
        pooled, transformed, features, transformed_targets, shapes = self.encoded(images, targets)
        logits = self.classifier(pooled)
        if self.training:
            if targets is None:
                raise ValueError("Training requires image labels and detection validity")
            selected = [i for i,t in enumerate(targets) if bool(t["detection_valid"])]
            losses = {}
            if self.task == "joint" and selected:
                ix = torch.tensor(selected, device=pooled.device)
                det_images = ImageList(transformed.tensors[ix], [transformed.image_sizes[i] for i in selected])
                det_features = {k:v[ix] for k,v in features.items()}
                det_targets = [transformed_targets[i] for i in selected]
                proposals, proposal_losses = self.detector.rpn(det_images,det_features,det_targets)
                _, roi_losses = self.detector.roi_heads(det_features,proposals,det_images.image_sizes,det_targets)
                losses = {**proposal_losses, **roi_losses}
            return {"logits":logits,"detection_losses":losses,"features":pooled}
        if self.task == "joint":
            proposals,_=self.detector.rpn(transformed,features,None)
            boxes,_=self.detector.roi_heads(features,proposals,transformed.image_sizes,None)
            boxes=self.detector.transform.postprocess(boxes,transformed.image_sizes,shapes)
        else:
            boxes=[{"boxes":torch.empty((0,4),device=pooled.device),
                    "scores":torch.empty(0,device=pooled.device),
                    "labels":torch.empty(0,dtype=torch.long,device=pooled.device)} for _ in images]
        return {"logits":logits,"detections":boxes,"features":pooled}
