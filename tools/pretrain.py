from ultralytics import YOLO

def main():
    for model_type in ["yolov8s_GD_CAFM_FMF_AConv"]:
        # print("Training "+model_type+":")
        # model = YOLO(model_type+".yaml").load(r"ChickenCounting\yolov8s_GD_CAFM_FMF_AConv_100e_coco\weights\last.pt")
        # model.train(data="coco.yaml", epochs=100, batch=-1, cache=False, project='ChickenCounting', name=model_type+"_200e_coco", plots=False)

        print("Resume training "+model_type+":")
        model = YOLO(f"ChickenCounting/{model_type}_200e_coco/weights/last.pt")
        model.train(resume=True)
        
if __name__ == "__main__":
    main()