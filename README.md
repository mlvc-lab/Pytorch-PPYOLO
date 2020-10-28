# Pytorch-PPYOLO

## Introduction
PPYOLO implemented in Pytorch
```
pytorch 1.5.0
torchvision 0.6.0
```

Original Paper(PaddlePaddle): https://github.com/PaddlePaddle/PaddleDetection
This Repo is forked from https://github.com/miemie2013/Pytorch-PPYOLO

## Sources
Keras YOLOv3: https://github.com/miemie2013/Keras-DIOU-YOLOv3

Pytorch YOLOv3: https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

PaddlePaddle YOLOv3: https://github.com/miemie2013/Paddle-DIOU-YOLOv3

PaddlePaddle yolact: https://github.com/miemie2013/PaddlePaddle_yolact

Keras YOLOv4: https://github.com/miemie2013/Keras-YOLOv4 (mAP 41%+)

Pytorch YOLOv4: https://github.com/miemie2013/Pytorch-YOLOv4 (mAP 41%+)

Paddle YOLOv4: https://github.com/miemie2013/Paddle-YOLOv4 (mAP 41%+)

PaddleDetection SOLOv2: https://github.com/miemie2013/PaddleDetection-SOLOv2

Pytorch FCOS YOLOv4: https://github.com/miemie2013/Pytorch-FCOS

Paddle FCOS YOLOv4: https://github.com/miemie2013/Paddle-FCOS

Keras CartoonGAN: https://github.com/miemie2013/keras_CartoonGAN

Pytorch PPYOLO: https://github.com/miemie2013/Pytorch-PPYOLO (mAP 44.8%)

## 실현 된 부분
EMA (지수 이동 평균) : config / ppyolo_2x.py에서 self.use_ema = True를 수정하여 엽니 다. 닫으려면 config / ppyolo_2x.py에서 self.use_ema = False를 수정하십시오.

DropBlock : 피처 맵에 무작위로 픽셀을 드롭합니다.

IoU 손실 : Iou 손실.

IoU Aware : 예측 상자와 gt의 iou를 예측합니다. 그리고 그것은 objness에 작용합니다.

그리드 감지 : 예측 상자 중심점의 xy는 그리드를 벗어날 수 있으며 gt 중심점은 그리드 선에 있습니다.

Matrix NMS : SOLOv2에서 제안한 알고리즘은 soft-nm 등을 기반으로 병렬화되고 가속화됩니다. 예측 상자에 유사한 높은 점수 상자가있는 경우 직접 버리는 대신 예측 상자의 점수를 낮 춥니 다. 여기서는 mask iou 대신 box iou를 사용합니다.

CoordConv : 기능 맵에있는 픽셀의 좌표 정보 (채널 수 + 2).

SPP : 3 개의 풀링 레이어와 원본 이미지 스티칭의 출력.

## Environment Setup

Install DCNv2
```
cd external/DCNv2
python setup.py build develop
```

Install requirements
```
pip install -r requirements.txt
apt update -y
apt install libgl1-mesa-glx -y
apt-get install libglib2.0-0 -y
```

## Train from Scratch
Simply run
```
python train.py
```
It will train new model based on configs defined in config/ppyolo_2x.py

##Train using Pretrained Model
下载我从PaddleDetection的仓库保存下来的pytorch模型ppyolo_2x.pt
链接：https://pan.baidu.com/s/18ZUQMWF7qPJ7K7xqx1VnpQ 
提取码：6hph 

该模型在COCO2017 val集的mAP如下
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.649
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.492
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.483
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.571
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.624
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.420
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.665
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773
```

##validation
python eval.py

##test
python demo.py
