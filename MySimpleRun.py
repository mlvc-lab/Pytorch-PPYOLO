import argparse
import torch
from torch.autograd import Variable
import time

from config import select_backbone, select_head, select_loss, PPYOLO_2x_Config
from model.ppyolo import PPYOLO
from tools.voc_dataloader import ListDataset


parser = argparse.ArgumentParser(description='PPYOLO Training Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=0,
                    choices=[0, 1],
                    help='0 -- ppyolo_2x.py;  1 -- ppyolo_1x.py;  ')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu


def build_model(cfg, loss):
    # define backbone
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)

    # define head
    Head = select_head(cfg.head_type)
    head = Head(yolo_loss=loss, is_train=True, nms_cfg=cfg.nms_cfg, **cfg.head)

    model = PPYOLO(backbone, head)

    return model

if __name__ == '__main__':
    cfg = None
    cfg = PPYOLO_2x_Config()

    #define loss
    IouLoss = select_loss(cfg.iou_loss_type)
    iou_loss = IouLoss(**cfg.iou_loss)
    IouAwareLoss = select_loss(cfg.iou_aware_loss_type)
    iou_aware_loss = IouAwareLoss(**cfg.iou_aware_loss)
    Loss = select_loss(cfg.yolo_loss_type)
    yolo_loss = Loss(iou_loss=iou_loss, iou_aware_loss=iou_aware_loss, **cfg.yolo_loss)

    #build model
    model = build_model(cfg, loss=yolo_loss)
    if use_gpu:
        model = model.cuda()

    #define optimizier
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=cfg.train_cfg['lr'])  # requires_grad==True 的参数才可以被更新

    # Get dataloader
    batch_size = cfg.train_cfg['batch_size']
    dataset = ListDataset(cfg.train_path, cfg.anno_path, cfg.im_path, augment=True, multiscale=cfg.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    for epoch in range(cfg.train_cfg['max_iters']):
        model.train()
        start_time = time.time()
        print("ah")
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.cuda())
            targets = Variable(targets.cuda(), requires_grad=False)

            loss = model(imgs, targets)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()