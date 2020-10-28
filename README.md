# Pytorch-PPYOLO

## Introductionฐ
PPYOLO implemented in Pytorch
```
pytorch 1.5.0
torchvision 0.6.0
```

Original Paper(PaddlePaddle): https://github.com/PaddlePaddle/PaddleDetection
This Repo is forked from https://github.com/miemie2013/Pytorch-PPYOLO

## Sources
Keras YOLOv3: https://github.com/miemie2013/Keras-DIOU-YOLOv3

Pytorch YOLOv3๏: https://github.com/miemie2013/Pytorch-DIOU-YOLOv3

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

## ์คํ ๋ ๋ถ๋ถ
EMA (์ง์ ์ด๋ ํ๊ท ) : config / ppyolo_2x.py์์ self.use_ema = True๋ฅผ ์์ ํ์ฌ ์ฝ๋ ๋ค. ๋ซ์ผ๋ ค๋ฉด config / ppyolo_2x.py์์ self.use_ema = False๋ฅผ ์์ ํ์ญ์์ค.

DropBlock : ํผ์ฒ ๋งต์ ๋ฌด์์๋ก ํฝ์์ ๋๋กญํฉ๋๋ค.

IoU ์์ค : Iou ์์ค.

IoU Aware : ์์ธก ์์์ gt์ iou๋ฅผ ์์ธกํฉ๋๋ค. ๊ทธ๋ฆฌ๊ณ  ๊ทธ๊ฒ์ objness์ ์์ฉํฉ๋๋ค.

๊ทธ๋ฆฌ๋ ๊ฐ์ง : ์์ธก ์์ ์ค์ฌ์ ์ xy๋ ๊ทธ๋ฆฌ๋๋ฅผ ๋ฒ์ด๋  ์ ์์ผ๋ฉฐ gt ์ค์ฌ์ ์ ๊ทธ๋ฆฌ๋ ์ ์ ์์ต๋๋ค.

Matrix NMS : SOLOv2์์ ์ ์ํ ์๊ณ ๋ฆฌ์ฆ์ soft-nm ๋ฑ์ ๊ธฐ๋ฐ์ผ๋ก ๋ณ๋ ฌํ๋๊ณ  ๊ฐ์ํ๋ฉ๋๋ค. ์์ธก ์์์ ์ ์ฌํ ๋์ ์ ์ ์์๊ฐ์๋ ๊ฒฝ์ฐ ์ง์  ๋ฒ๋ฆฌ๋ ๋์  ์์ธก ์์์ ์ ์๋ฅผ ๋ฎ ์ถฅ๋ ๋ค. ์ฌ๊ธฐ์๋ mask iou ๋์  box iou๋ฅผ ์ฌ์ฉํฉ๋๋ค.

CoordConv : ๊ธฐ๋ฅ ๋งต์์๋ ํฝ์์ ์ขํ ์ ๋ณด (์ฑ๋ ์ + 2).

SPP : 3 ๊ฐ์ ํ๋ง ๋ ์ด์ด์ ์๋ณธ ์ด๋ฏธ์ง ์คํฐ์นญ์ ์ถ๋ ฅ.

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
ไธ่ฝฝๆไปPaddleDetection็ไปๅบไฟๅญไธๆฅ็pytorchๆจกๅppyolo_2x.pt
้พๆฅ๏ผhttps://pan.baidu.com/s/18ZUQMWF7qPJ7K7xqx1VnpQ 
ๆๅ็ ๏ผ6hph 

่ฏฅๆจกๅๅจCOCO2017 val้็mAPๅฆไธ
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
