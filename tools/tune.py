from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolov8s_MCWA_AConv_C2INXB_GD.yaml").load(r"ChickenCounting/yolov8s_MCWA_AConv_C2INXB_GD_100e_coco/weights/last.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data="./data/data.yaml", epochs=30, iterations=300, optimizer="AdamW", plots=False, save=False, val=False, iou_type='WIoUv2',)