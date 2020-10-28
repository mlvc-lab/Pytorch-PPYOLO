#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-10-15 14:50:03
#   Description : pytorch_ppyolo
#
# ================================================================



class PPYOLO_2x_Config(object):
    def __init__(self):
        # 自定义数据集
        # self.train_path = 'annotation_json/voc2012_train.json'
        # self.val_path = 'annotation_json/voc2012_val.json'
        # self.classes_path = 'data/voc_classes.txt'
        # self.train_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
        # self.val_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径

        # COCO
        '''
        self.train_path = '/dataset/coco/annotations/instances_train2017.json'
        self.val_path = '/dataset/coco/annotations/instances_val2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.train_pre_path = '/dataset/coco/images/train2017/'  # 训练集图片相对路径
        self.val_pre_path = '/dataset/coco/images/val2017/'      # 验证集图片相对路径
        '''

        # PASCALVOC
        self.train_path = 'voc_train2017.json' #'/dataset/VOC0712/VOC2012/ImageSets/Main/train.txt'
        self.val_path = 'voc_test2017.json' #'/dataset/VOC0712/VOC2012/ImageSets/Main/val.txt'
        self.classes_path = 'data/voc_classes.txt'
        self.train_pre_path = '/dataset/VOC0712/VOC2012/JPEGImages/'
        self.val_pre_path = '/dataset/VOC0712/VOC2012/JPEGImages/'

        #for MySimpleRun
        self.anno_path = '/dataset/VOC0712/VOC2012/Annotations'
        self.im_path = '/dataset/VOC0712/VOC2012/JPEGImages'
        self.multiscale_training = False
        self.num_workers = 0
        # ========= 一些设置 =========
        self.train_cfg = dict(
            lr=0.001,
            batch_size=8,
            model_path=None,
            # model_path='./weights/step00005000.pt',
            save_iter=2500,
            eval_iter=5000,
            max_iters=500000,
            multi_gpus=True,
        )

        image_size = 608 #608 in original repo
        num_classes = 20

        # 验证。用于train.py、eval.py、test_dev.py
        self.eval_cfg = dict(
            model_path='weights/step00100000.pt',
            # model_path='./weights/step00005000.pt',
            target_size=image_size,
            draw_image=False,    # 是否画出验证集图片
            draw_thresh=0.15,    # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
            eval_batch_size=1,   # 验证时的批大小。由于太麻烦，暂时只支持1。
        )

        # 测试。用于demo.py
        self.test_cfg = dict(
            model_path='ppyolo_2x.pt',
            # model_path='./weights/step00010000.pt',
            target_size=image_size,
            draw_image=True,
            draw_thresh=0.15,   # 如果draw_image==True，那么只画出分数超过draw_thresh的物体的预测框。
        )


        # ============= 模型相关 =============
        self.backbone_type = 'Resnet50Vd'
        self.backbone = dict(
            norm_type='bn',
            feature_maps=[3, 4, 5],
            dcn_v2_stages=[5],
            downsample_in3x3=True,   # 注意这个细节，是在3x3卷积层下采样的。
        )
        self.head_type = 'YOLOv3Head'
        self.head = dict(
            num_classes=num_classes,
            # num_classes=20,
            norm_type='bn',
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            anchors=[[10, 13], [16, 30], [33, 23],
                     [30, 61], [62, 45], [59, 119],
                     [116, 90], [156, 198], [373, 326]],
            coord_conv=True,
            iou_aware=True,
            iou_aware_factor=0.4,
            scale_x_y=1.05,
            spp=True,
            drop_block=True,
            downsample=[32, 16, 8],
            in_channels=[2048, 1024, 512],
        )
        self.iou_loss_type = 'IouLoss'
        self.iou_loss = dict(
            loss_weight=2.5,
            max_height=image_size,
            max_width=image_size,
            ciou_term=False,
        )
        self.iou_aware_loss_type = 'IouAwareLoss'
        self.iou_aware_loss = dict(
            loss_weight=1.0,
            max_height=image_size,
            max_width=image_size,
        )
        self.yolo_loss_type = 'YOLOv3Loss'
        self.yolo_loss = dict(
            ignore_thresh=0.7,
            scale_x_y=1.05,
            label_smooth=False,
            use_fine_grained_loss=True,
        )
        self.nms_cfg = dict(
            nms_type='matrix_nms',
            score_threshold=0.01,
            post_threshold=0.01,
            nms_top_k=-1,
            keep_top_k=100,
            use_gaussian=False,
            gaussian_sigma=2.,
        )


        # ============= 预处理相关 =============
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # DecodeImage
        self.decodeImage = dict(
            to_rgb=True,
            with_mixup=True, #true in original repo
        )
        # MixupImage
        self.mixupImage = dict(
            alpha=1.5,
            beta=1.5,
        )
        # ColorDistort
        self.colorDistort = dict()
        # RandomExpand
        self.randomExpand = dict(
            fill_value=[123.675, 116.28, 103.53],
        )
        # RandomCrop
        self.randomCrop = dict()
        # RandomFlipImage
        self.randomFlipImage = dict(
            is_normalized=False,
        )
        # NormalizeBox
        self.normalizeBox = dict()
        # PadBox
        self.padBox = dict(
            num_max_boxes=50,
        )
        # BboxXYXY2XYWH
        self.bboxXYXY2XYWH = dict()
        # RandomShape
        self.randomShape = dict(
            sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
            random_inter=True, #True in original repo
        )
        # NormalizeImage
        self.normalizeImage = dict(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True,
            is_channel_first=False,
        )
        # Permute
        self.permute = dict(
            to_bgr=False,
            channel_first=True,
        )
        # Gt2YoloTarget
        self.gt2YoloTarget = dict(
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            anchors=[[10, 13], [16, 30], [33, 23],
                     [30, 61], [62, 45], [59, 119],
                     [116, 90], [156, 198], [373, 326]],
            downsample_ratios=[32, 16, 8],
            num_classes=num_classes,
        )
        # ResizeImage
        self.resizeImage = dict(
            target_size=image_size,
            interp=2,
        )


