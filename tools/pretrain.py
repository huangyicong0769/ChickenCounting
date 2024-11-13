from ultralytics import YOLO

def main():
    for model_type in ["yolov8s_MCA_C2INXB_GD"]:
        print("Training "+model_type+":")
        model = YOLO(model_type+".yaml")
        model.train(data="coco.yaml", epochs=100, batch=-1, cache=False, project='ChickenCounting', name=model_type+"_100e_coco", plots=False)

    # print("Resume training "+model_type+":")
    # model = YOLO("baseline/yolov8s_CBAM_500e_coco3/weights/last.pt")
    # model.train(resume=True, cache=False)
        
if __name__ == "__main__":
    main()