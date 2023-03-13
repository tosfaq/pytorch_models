from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models
from re import sub
from os.path import basename, join, exists, splitext, basename
from glob import glob
from PIL import Image
import albumentations as A
import numpy as np
import cv2
import os
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics.classification import BinaryJaccardIndex
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


import matplotlib.pyplot as plt

class BirdsSegDataset(Dataset):
    def __init__(self,
                 img_dir,
                 gt_dir,
                 transform=None
                 ):
        
        self._input_images = glob(join(img_dir, '**/*.jpg'))
        self._target_masks = list(map(lambda x: sub('.*(/.*/.*)\.jpg', gt_dir+'\g<1>.png', x), self._input_images))
        
        self._transform = transform
        self.transformations_mask = transforms.Compose([
                transforms.Resize((224,224)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        self.transformations_alb = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
        self.transformations_alb_mask = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
        self.transformations = transforms.Compose([
                transforms.Resize((224,224)),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])

    def __len__(self):
        return len(self._input_images)
    
    def __getitem__(self, index):
        img_path, mask_path = self._input_images[index], self._target_masks[index]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        ## augmentation
        if self._transform is not None:
            transformed = self._transform(image=np.array(image), mask=np.array(mask))
            image, mask = transformed['image'], transformed['mask']
            image = self.transformations_alb(image)
            mask = self.transformations_alb_mask(mask)
        else:
            image = self.transformations(image)
            mask = self.transformations_mask(mask)
        
        return image, mask[0].unsqueeze(0)



def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


class MobileNetV2UNet(nn.Module):
    def __init__(self, n_class, weights):
        super().__init__()
        # weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
        self.base_model = torchvision.models.mobilenet_v2(weights=weights)


        self.base_layers = list(self.base_model.children())[0]

        self.layer0 = nn.Sequential(*self.base_layers[:2]) # size=(N, 16, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(16, 16, 1, 0)
        self.layer1 = self.base_layers[2] # size=(N, 24, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(24, 24, 1, 0)
        self.layer2 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 32, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(32, 32, 1, 0)
        self.layer3 = nn.Sequential(*self.base_layers[5:8])  # size=(N, 64, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(64, 64, 1, 0)
        self.layer4 = nn.Sequential(*self.base_layers[8:18])  # size=(N, 320, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(320, 320, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(64 + 320, 320, 3, 1)
        self.conv_up2 = convrelu(32 + 320, 64, 3, 1)
        self.conv_up1 = convrelu(24 + 64, 64, 3, 1)
        self.conv_up0 = convrelu(16 + 64, 32, 3, 1)

        self.conv_original_size0 = convrelu(3, 32, 3, 1)
        self.conv_original_size1 = convrelu(32, 16, 3, 1)
        self.conv_original_size2 = convrelu(16 + 32, 16, 3, 1)

        self.conv_last = nn.Conv2d(16, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +\
                                                 target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class MyModel(pl.LightningModule):
    def __init__(self, num_classes, weights = None):
        super().__init__()

        self.model = MobileNetV2UNet(num_classes, weights)
        
        # freeze backbone layers
        for l in self.model.base_layers:
            for param in l.parameters():
                param.requires_grad = False
        
        self.metric = BinaryJaccardIndex(threshold=0.5)
        
        self.bce_weight = 0.9
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """the full training loop"""
        x, y = batch

        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)
        
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)

        iou = self.metric(y, (pred > 0.5).float())

        return {'loss': loss, 'iou': iou}
    
    def configure_optimizers(self):
        """ Define optimizers and LR schedulers. """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=5e-4)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0 = 50,
                                                                            verbose=True)
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss"
        } 
        
        return [optimizer], [lr_dict]
    
    def validation_step(self, batch, batch_idx):
        """the full validation loop"""
        x, y = batch

        y_logit = self(x)        
        bce = F.binary_cross_entropy_with_logits(y_logit, y)
        
        pred = torch.sigmoid(y_logit)
        dice = dice_loss(pred, y)
        
        loss = bce * self.bce_weight + dice * (1 - self.bce_weight) * y.size(0)

        iou = self.metric(y, (pred > 0.5).float())

        return {'val_loss': loss, 'logs':{'dice':dice, 'bce': bce, 'iou': iou}}

    def training_epoch_end(self, outputs):
        """log and display average train loss and accuracy across epoch"""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        
        print(f"| Train_loss: {avg_loss:.3f}, Train_iou: {avg_iou:.3f}" )
        self.log('train_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)
     
    def validation_epoch_end(self, outputs):
        """log and display average val loss and accuracy"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        avg_dice = torch.stack([x['logs']['dice'] for x in outputs]).mean()
        avg_bce = torch.stack([x['logs']['bce'] for x in outputs]).mean()
        avg_iou = torch.stack([x['logs']['iou'] for x in outputs]).mean()
        
        print(f"[Epoch {self.trainer.current_epoch:3}] Val_IoU: {avg_iou:.3f}, Val_loss: {avg_loss:.3f}, Val_dice: {avg_dice:.3f}, Val_bce: {avg_bce:.3f}", end= " ")
        self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True, on_step=False)



def predict(model, img_path):
    """
    Функция predict(model, img_path) предсказывает карту вероятностей класса фон-объект для изображения. 
    Карта вероятностей — это двумерная матрица размера H × W , где H и W — высота и ширина сегментируемого изображения. 
    В каждой ячейке такой матрицы записано вещественное число от 0 до 1 — вероятность принадлежности соответствующего пикселя изображения объекту. 
    """
    model.eval()
    image = Image.open(img_path).convert('RGB')
    initial_shape = np.array(image).shape[:2]
    prepare = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    image = prepare(image)
    pred = torch.sigmoid(model(image.unsqueeze(0))[0])
    result = cv2.resize(pred[0].detach().numpy(),initial_shape[::-1])
    return result


def get_model():
    return MyModel(num_classes = 1, weights = None)

def train_model(train_data_path):
    transformations = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ]
    )
    
    train_set = BirdsSegDataset(img_dir=os.path.join(train_data_path, 'images'), gt_dir=os.path.join(train_data_path, 'gt'), transform = transformations)
    #Init Dataloaders
    dl_train = DataLoader(train_set, batch_size=32, shuffle=True)

    DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator=DEVICE,
        devices=1,
        callbacks = [], 
        logger = False, 
        enable_checkpointing=False,
    )

    model = MyModel(num_classes = 1)
    trainer.fit(model, dl_train)
    torch.save(model.state_dict(), 'segmentation_model.pth')

    