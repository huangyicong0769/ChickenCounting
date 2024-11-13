# from os.path import exists

from ultralytics import YOLO

def main():
        model_type = "yolov8s_C2fS_C2INXB_GD"
        model = YOLO(model_type+".yaml")
        model.train(data='./data/data.yaml', epochs=5, batch=16, cache=True, pretrained=False, save=False, plots=False)
        
if __name__ == "__main__":
    main()