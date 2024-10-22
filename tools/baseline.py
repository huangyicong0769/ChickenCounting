from ultralytics import YOLO

v8s = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

def main():
    for model_type in ["yolov8s", "yolov9s", "yolov10s", "yolo11s"]:
        print("Training "+model_type+" baseline:")
        model = YOLO(model_type+".yaml")
        model.train(data='./data/data.yaml', epochs=350, batch=0.80, cache=True, project='ChickenCounting', name=model_type+"_350e_noPretrained", plots=True, pretrained=False)
        model.val()
        
if __name__ == "__main__":
    main()