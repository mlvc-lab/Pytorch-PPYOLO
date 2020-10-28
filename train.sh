#pip uninstall torch -y
#conda create -n torch14 python=3.7.7
#conda activate torch14
#conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

cd /root/volume/Pytorch-PPYOLO/
cd external/DCNv2
python setup.py build develop

cd /root/volume/Pytorch-PPYOLO/

pip install -r requirements.txt
apt update -y
apt install libgl1-mesa-glx -y
apt-get install libglib2.0-0 -y

pip list

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py
