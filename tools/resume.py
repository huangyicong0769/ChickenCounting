# from os.path import exists

from ultralytics import YOLO

# pretrain = True

def main():
    model_type = "yolov8s_MCA_C2INXB_GD"
    print(f"Resuming {model_type}:")
    model = YOLO(r"ChickenCounting\yolov8s_MCA_C2INXB_GD_cocopt4\weights\last.pt")
    model.train(resume=True)
        
if __name__ == "__main__":
    main()