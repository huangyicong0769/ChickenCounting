# from os.path import exists

from ultralytics import YOLO

def main():
        model_type = "yolov8s_MCA_C2INXB_LSKGD_ii"
        model = YOLO(model_type+".yaml")
        # model = YOLO("ChickenCounting/yolov8s_MCA_AConv_C2PSA_GD_100e_coco2/weights/best.pt")
        model.train(data='./data/data.yaml', epochs=25, batch=0.80, cache=True, pretrained=False, save=False, plots=False)
        
if __name__ == "__main__":
    main()