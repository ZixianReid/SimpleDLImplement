vgg16_pretrain_path = "./checkpoints/vgg16-397923af.pth"

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


voc_data_dir = '/mnt/data/VOCdevkit/VOC2007'
min_size = 600  # image resize
max_size = 1000 # image resize
num_workers = 8
test_num_workers = 8