# from os.path import exists

from ultralytics import YOLO

def main():
        model_type = "yolov8s_S2fMCA_AConv_C2INXB_GD"
        model = YOLO(model_type+".yaml")
        model.train(data='./data/data.yaml', epochs=25, batch=32, cache=True, pretrained=False, save=False, plots=False)
        
if __name__ == "__main__":
    main()