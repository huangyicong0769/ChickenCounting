from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO(r"ChickenCounting\yolov8s_GD_xi_200e_\weights\best.pt")  # load a custom trained model

    # Export the model
    model.export(format="onnx")

if __name__ == '__main__':
    main()