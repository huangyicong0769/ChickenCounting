# from os.path import exists

from ultralytics import YOLO

# pretrain = True

def main():
    model_type = "yolov8s_S2fMCA_AConv_C2INXB_GD"
    print(f"Resuming {model_type}:")
    model = YOLO(r"ChickenCounting\yolov8s_S2fMCA_AConv_C2INXB_GD_\weights\last.pt")
    model.train(resume=True)
        
if __name__ == "__main__":
    main()