from ray import tune
from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolov8s_MCWA_AConv_C2INXB_GD.yaml").load(r"ChickenCounting/yolov8s_MCWA_AConv_C2INXB_GD_100e_coco/weights/last.pt")

space = {  # key: (min, max, gain(optional))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": tune.uniform(1e-5, 5e-3),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": tune.uniform(0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": tune.uniform(0.7, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": tune.uniform(0.0, 0.95),  # warmup initial momentum
            "box": tune.uniform(1.0, 20.0),  # box loss gain
            "cls": tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
            "dfl": tune.uniform(0.4, 6.0),  # dfl loss gain
            "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
            # "degrees": (0.0, 45.0),  # image rotation (+/- deg)
            "translate": tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
            "scale": tune.uniform(0.0, 0.95),  # image scale (+/- gain)
            # "shear": (0.0, 10.0),  # image shear (+/- deg)
            # "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            # "flipud": (0.0, 1.0),  # image flip up-down (probability)
            "fliplr": tune.uniform(0.0, 1.0),  # image flip left-right (probability)
            # "bgr": (0.0, 1.0),  # image channel bgr (probability)
            "mosaic": tune.uniform(0.0, 1.0),  # image mixup (probability)
            # "mixup": (0.0, 1.0),  # image mixup (probability)
            # "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
            # "ki": (0.0, 10.0),
            # "kp": (0.0, 2.0),
            "close_mosaic": tune.quniform(0, 30),
        }

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data="./data/data.yaml", epochs=50, iterations=300, optimizer="Lion", plots=False, save=False, val=False, use_ray=True, space=space, use_ray=True,)