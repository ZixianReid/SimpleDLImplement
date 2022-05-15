# SimpleDLImplement

## Environment
Python 3.8, CUDA 11.3, cuDNN 8, NVCC, Pytorch 1.11.0, torchvision 0.12.0, torchaudio 0.11.0\


## For starter to learn
Try to implement deeping learning network with the flatten way. \
Try to implement deeping learning network with least packages.


## Data preparing

###  

### ModelNet40
Download http://modelnet.cs.princeton.edu/ \
please refer https://github.com/charlesq34/pointnet for details

### VOC2007  
please refer https://github.com/rbgirshick/py-faster-rcnn for details

## Network implemented

### pointnet for classification
network in point_cls.py \
train in main.py

### pointnet++ for classification
network in pointplus_cls.py \
train in main.py

### Fast-RCNN for object detection
still deveoloping


### Todo

- [ ] Fast-RCNN implementation
- [ ] Pointpillar  implementation
- [ ] YOLO5 implementation


## Reference
https://github.com/charlesq34/pointnet \
https://github.com/chenyuntc/simple-faster-rcnn-pytorch \
https://github.com/erikwijmans/Pointnet2_PyTorch