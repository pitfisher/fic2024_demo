import torch
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = YOLO("yolo11n.pt")
results = model.train(
    data="mdata.yaml", augment=False, device=[0,1],
    epochs=500,
    batch=72, 
    workers=16, 
    imgsz=1024,
    optimizer='RAdam',
    dropout=0.25,
    lr0=0.01,
    lrf=0.0001,
    cos_lr=True,
    hsv_h= 0.015,
    hsv_s= 0.7,
    hsv_v= 0.4,
    degrees= 0.15,
    translate= 0.1,
    scale= 0.1,
    shear= 0.05,
    perspective=0.0,
    flipud= 0.5,
    fliplr= 0.5,
    bgr= 0.1,
    mosaic= 0.75,
    mixup= 0.2,
    copy_paste= 0.2,
    auto_augment= 'augmix',
    erasing= 0.4,
    crop_fraction= 0.25,
    name='train_v11_1024_n_all',
    )