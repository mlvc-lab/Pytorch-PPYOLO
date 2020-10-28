#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================
from collections import deque
import time
import threading
import datetime
from collections import OrderedDict
import os
import argparse

from config import *

from model.ppyolo import PPYOLO
from tools.cocotools import get_classes, catid2clsid, clsid2catid
from model.decode_np import Decode
from tools.cocotools import eval
from tools.data_process import data_clean, get_samples
from tools.transform import *
from pycocotools.coco import COCO

import pytorch_warmup as warmup

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PPYOLO Training Script')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--config', type=int, default=0,
                    choices=[0, 1],
                    help='0 -- ppyolo_2x.py;  1 -- ppyolo_1x.py;  ')
args = parser.parse_args()
config_file = args.config
use_gpu = args.use_gpu


import platform
sysstr = platform.system()
print(torch.cuda.is_available())
print(torch.__version__)
# 禁用cudnn就能解决Windows报错问题。Windows用户如果删掉之后不报CUDNN_STATUS_EXECUTION_FAILED，那就可以删掉。
if sysstr == 'Windows':
    torch.backends.cudnn.enabled = False



def multi_thread_op(i, samples, decodeImage, context, with_mixup, mixupImage, colorDistort,
                           randomExpand, randomCrop, randomFlipImage, normalizeBox, padBox, bboxXYXY2XYWH):
    samples[i] = decodeImage(samples[i], context)
    if with_mixup:
        samples[i] = mixupImage(samples[i], context)
    samples[i] = colorDistort(samples[i], context)
    samples[i] = randomExpand(samples[i], context)
    samples[i] = randomCrop(samples[i], context)
    samples[i] = randomFlipImage(samples[i], context)
    samples[i] = normalizeBox(samples[i], context)
    samples[i] = padBox(samples[i], context)
    samples[i] = bboxXYXY2XYWH(samples[i], context)

def load_weights(model, model_path):
    _state_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if k in _state_dict:
            shape_1 = _state_dict[k].shape
            shape_2 = pretrained_dict[k].shape
            if shape_1 == shape_2:
                new_state_dict[k] = v
            else:
                print('shape mismatch in %s. shape_1=%s, while shape_2=%s.' % (k, shape_1, shape_2))
    _state_dict.update(new_state_dict)
    model.load_state_dict(_state_dict)


if __name__ == '__main__':
    cfg = None
    if config_file == 0:
        cfg = PPYOLO_2x_Config()
    elif config_file == 1:
        cfg = PPYOLO_2x_Config()

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)

    # 步id，无需设置，会自动读。
    iter_id = 0

    # define loss
    IouLoss = select_loss(cfg.iou_loss_type)
    iou_loss = IouLoss(**cfg.iou_loss)
    IouAwareLoss = select_loss(cfg.iou_aware_loss_type)
    iou_aware_loss = IouAwareLoss(**cfg.iou_aware_loss)
    Loss = select_loss(cfg.yolo_loss_type)
    yolo_loss = Loss(iou_loss=iou_loss, iou_aware_loss=iou_aware_loss, **cfg.yolo_loss)

    #define model
    Backbone = select_backbone(cfg.backbone_type)
    backbone = Backbone(**cfg.backbone)
    Head = select_head(cfg.head_type)
    head = Head(yolo_loss=yolo_loss, is_train=True, nms_cfg=cfg.nms_cfg, **cfg.head)
    ppyolo = PPYOLO(backbone, head)

    #look into decode class
    _decode = Decode(ppyolo, class_names, use_gpu, cfg, for_test=False)

    # 加载权重
    if cfg.train_cfg['model_path'] is not None:
        # 加载参数, 跳过形状不匹配的。
        load_weights(ppyolo, cfg.train_cfg['model_path'])

        strs = cfg.train_cfg['model_path'].split('step')
        if len(strs) == 2:
            iter_id = int(strs[1][:8])

        # 冻结，使得需要的显存减少。低显存的卡建议这样配置。
        if cfg.backbone_type == 'Resnet50Vd':
            backbone.freeze(freeze_at=5)


    if use_gpu:   # 如果有gpu可用，模型（包括了权重weight）存放在gpu显存里
        ppyolo = ppyolo.cuda()

    # 种类id
    _catid2clsid = copy.deepcopy(catid2clsid)
    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _catid2clsid = {}
        _clsid2catid = {}
        for k in range(num_classes):
            _catid2clsid[k] = k
            _clsid2catid[k] = k
    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    # 验证集
    val_dataset = COCO(cfg.val_path)
    val_img_ids = val_dataset.getImgIds()
    val_images = []   # 只跑有gt的图片，跟随PaddleDetection
    for img_id in val_img_ids:
        ins_anno_ids = val_dataset.getAnnIds(imgIds=img_id, iscrowd=False)   # 读取这张图片所有标注anno的id
        if len(ins_anno_ids) == 0:
            continue
        img_anno = val_dataset.loadImgs(img_id)[0]
        val_images.append(img_anno)

    batch_size = cfg.train_cfg['batch_size']
    with_mixup = cfg.decodeImage['with_mixup']
    context = cfg.context
    # 预处理
    # sample_transforms
    decodeImage = DecodeImage(**cfg.decodeImage)   # 对图片解码。最开始的一步。
    mixupImage = MixupImage(**cfg.mixupImage)      # mixup增强
    colorDistort = ColorDistort(**cfg.colorDistort)  # 颜色扰动
    randomExpand = RandomExpand(**cfg.randomExpand)  # 随机填充
    randomCrop = RandomCrop(**cfg.randomCrop)        # 随机裁剪
    randomFlipImage = RandomFlipImage(**cfg.randomFlipImage)  # 随机翻转
    normalizeBox = NormalizeBox(**cfg.normalizeBox)        # 将物体的左上角坐标、右下角坐标中的横坐标/图片宽、纵坐标/图片高 以归一化坐标。
    padBox = PadBox(**cfg.padBox)                          # 如果gt_bboxes的数量少于num_max_boxes，那么填充坐标是0的bboxes以凑够num_max_boxes。
    bboxXYXY2XYWH = BboxXYXY2XYWH(**cfg.bboxXYXY2XYWH)     # sample['gt_bbox']被改写为cx_cy_w_h格式。
    # batch_transforms
    randomShape = RandomShape(**cfg.randomShape)     # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
    normalizeImage = NormalizeImage(**cfg.normalizeImage)     # 图片归一化。先除以255归一化，再减均值除以标准差
    permute = Permute(**cfg.permute)    # 图片从HWC格式变成CHW格式
    gt2YoloTarget = Gt2YoloTarget(**cfg.gt2YoloTarget)   # 填写target张量。

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, ppyolo.parameters()), lr=cfg.train_cfg['lr'], momentum=0.9, weight_decay=0.0005)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, ppyolo.parameters()), lr=cfg.train_cfg['lr'])   # requires_grad==True 的参数才可以被更新

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500000,2000000], gamma=0.1)
    warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=1000)
    warmup_scheduler.last_step = -1
    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size
    best_ap_list = [0.0, 0]  #[map, iter]

    if use_gpu and cfg.train_cfg['multi_gpus']:
        print('using multi gpu.')
        print('__Number CUDA Devices:', torch.cuda.device_count())
        ppyolo = torch.nn.DataParallel(ppyolo)

    while True:   # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.train_cfg['max_iters'] - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)
            # sample_transforms用多线程
            threads = []
            for i in range(batch_size):
                t = threading.Thread(target=multi_thread_op, args=(i, samples, decodeImage, context, with_mixup, mixupImage, colorDistort,
                                                                   randomExpand, randomCrop, randomFlipImage, normalizeBox, padBox, bboxXYXY2XYWH))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            # batch_transforms
            samples = randomShape(samples, context)
            samples = normalizeImage(samples, context)
            samples = permute(samples, context)
            samples = gt2YoloTarget(samples, context)

            # 整理成ndarray
            images = []
            gt_bbox = []
            gt_score = []
            gt_class = []
            target0 = []
            target1 = []
            target2 = []
            for i in range(batch_size):
                sample = samples[i]
                images.append(np.expand_dims(sample['image'].astype(np.float32), 0))
                gt_bbox.append(np.expand_dims(sample['gt_bbox'].astype(np.float32), 0))
                gt_score.append(np.expand_dims(sample['gt_score'].astype(np.float32), 0))
                gt_class.append(np.expand_dims(sample['gt_class'].astype(np.int32), 0))
                target0.append(np.expand_dims(sample['target0'].astype(np.float32), 0))
                target1.append(np.expand_dims(sample['target1'].astype(np.float32), 0))
                target2.append(np.expand_dims(sample['target2'].astype(np.float32), 0))
            images = np.concatenate(images, 0)
            gt_bbox = np.concatenate(gt_bbox, 0)
            gt_score = np.concatenate(gt_score, 0)
            gt_class = np.concatenate(gt_class, 0)
            target0 = np.concatenate(target0, 0)
            target1 = np.concatenate(target1, 0)
            target2 = np.concatenate(target2, 0)

            images = torch.Tensor(images).contiguous()
            gt_bbox = torch.Tensor(gt_bbox).contiguous()
            gt_score = torch.Tensor(gt_score).contiguous()
            gt_class = torch.Tensor(gt_class).contiguous()
            target0 = torch.Tensor(target0).contiguous()
            target1 = torch.Tensor(target1).contiguous()
            target2 = torch.Tensor(target2).contiguous()

            if use_gpu:
                images = images.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_score = gt_score.cuda()
                gt_class = gt_class.cuda()
                target0 = target0.cuda()
                target1 = target1.cuda()
                target2 = target2.cuda()
            targets = [target0, target1, target2]
            losses = ppyolo(images, None, False, gt_bbox, gt_class, gt_score, targets)
            loss_xy = losses['loss_xy']
            loss_wh = losses['loss_wh']
            loss_obj = losses['loss_obj']
            loss_cls = losses['loss_cls']
            loss_iou = losses['loss_iou']
            loss_iou_aware = losses['loss_iou_aware']

            if use_gpu and cfg.train_cfg['multi_gpus']:
                mean_loss_xy = loss_xy.mean()
                mean_loss_wh = loss_wh.mean()
                mean_loss_obj = loss_obj.mean()
                mean_loss_cls = loss_cls.mean()
                mean_loss_iou = loss_iou.mean()
                mean_loss_iou_aware = loss_iou_aware.mean()

            if use_gpu and cfg.train_cfg['multi_gpus']:
                all_loss = mean_loss_xy + mean_loss_wh + mean_loss_obj + mean_loss_cls + mean_loss_iou + mean_loss_iou_aware
                _all_loss = all_loss.cpu().data.numpy()
                _loss_xy = mean_loss_xy.cpu().data.numpy()
                _loss_wh = mean_loss_wh.cpu().data.numpy()
                _loss_obj = mean_loss_obj.cpu().data.numpy()
                _loss_cls = mean_loss_cls.cpu().data.numpy()
                _loss_iou = mean_loss_iou.cpu().data.numpy()
                _loss_iou_aware = mean_loss_iou_aware.cpu().data.numpy()
            else:
                all_loss = loss_xy + loss_wh + loss_obj + loss_cls + loss_iou + loss_iou_aware
                _all_loss = all_loss.cpu().data.numpy()
                _loss_xy = loss_xy.cpu().data.numpy()
                _loss_wh = loss_wh.cpu().data.numpy()
                _loss_obj = loss_obj.cpu().data.numpy()
                _loss_cls = loss_cls.cpu().data.numpy()
                _loss_iou = loss_iou.cpu().data.numpy()
                _loss_iou_aware = loss_iou_aware.cpu().data.numpy()

            all_loss = all_loss.contiguous()

            # training step
            scheduler.step(train_steps-1)
            warmup_scheduler.dampen()
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            # ==================== log ====================
            if iter_id % 20 == 0:
                strs = 'Train iter: {}, all_loss: {:.6f}, loss_xy: {:.6f}, loss_wh: {:.6f}, loss_obj: {:.6f}, loss_cls: {:.6f}, loss_iou: {:.6f}, loss_iou_aware: {:.6f}, eta: {}'.format(
                    iter_id, _all_loss, _loss_xy, _loss_wh, _loss_obj, _loss_cls, _loss_iou, _loss_iou_aware, eta)
                print("LR: ", scheduler.get_lr())
                logger.info(strs)

            # ==================== save ====================
            if iter_id % cfg.train_cfg['save_iter'] == 0:
                save_path = './weights/step%.8d.pt' % iter_id
                torch.save(ppyolo.state_dict(), save_path)
                path_dir = os.listdir('./weights')
                steps = []
                names = []
                for name in path_dir:
                    if name[len(name) - 2:len(name)] == 'pt' and name[0:4] == 'step':
                        step = int(name[4:12])
                        steps.append(step)
                        names.append(name)
                if len(steps) > 10:
                    i = steps.index(min(steps))
                    os.remove('./weights/'+names[i])
                logger.info('Save model to {}'.format(save_path))

            # ==================== eval ====================
            if iter_id % cfg.train_cfg['eval_iter'] == 0:
                ppyolo.eval()   # 切换到验证模式
                box_ap = eval(_decode, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_cfg['eval_batch_size'], _clsid2catid, cfg.eval_cfg['draw_image'], cfg.eval_cfg['draw_thresh'])
                logger.info("box ap: %.3f" % (box_ap[0], ))
                ppyolo.train()  # 切换到训练模式

                # 以box_ap作为标准
                ap = box_ap
                if ap[0] > best_ap_list[0]:
                    best_ap_list[0] = ap[0]
                    best_ap_list[1] = iter_id
                    torch.save(ppyolo.state_dict(), './weights/best_model.pt')
                logger.info("Best test ap: {}, in iter: {}".format(best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.train_cfg['max_iters']:
                logger.info('Done.')
                exit(0)

