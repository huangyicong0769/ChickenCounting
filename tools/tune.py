from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolov8s_MCWA_AConv_C2INXB_GD.yaml").load(r"ChickenCounting/yolov8s_MCWA_AConv_C2INXB_GD_100e_coco/weights/last.pt")

space = {  # key: (min, max, gain(optional))
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": (1e-5, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            "lrf": (0.0001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (0.7, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (0.0, 0.001),  # optimizer weight decay 5e-4
            "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
            "box": (1.0, 20.0),  # box loss gain
            "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
            "dfl": (0.4, 6.0),  # dfl loss gain
            "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
            # "degrees": (0.0, 45.0),  # image rotation (+/- deg)
            "translate": (0.0, 0.9),  # image translation (+/- fraction)
            "scale": (0.0, 0.95),  # image scale (+/- gain)
            # "shear": (0.0, 10.0),  # image shear (+/- deg)
            # "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            # "flipud": (0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (0.0, 1.0),  # image flip left-right (probability)
            # "bgr": (0.0, 1.0),  # image channel bgr (probability)
            "mosaic": (0.0, 1.0),  # image mixup (probability)
            # "mixup": (0.0, 1.0),  # image mixup (probability)
            # "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
            "ki": (0.0, 10.0),
            "kd": (0.0, 2.0),
        }

# Tune hyperparameters for 30 epochs
model.tune(data=r"data/data.yaml", epochs=120, iterations=500, optimizer="PIDAO_ST", plots=False, save=False, val=False, space=space)