# from os.path import exists

from ultralytics import YOLO

def main():
    for model_type in ["yolov8s_MCA_C2INXB_GDSTA"]:
        print(f"Training {model_type}:")

        model = YOLO(model_type+".yaml")
        # model = YOLO("ChickenCounting/yolov8s_MCA_AConv_C2PSA_GD_100e_coco2/weights/best.pt")
        model.train(data='./data/data.yaml', epochs=350, batch=0.90, cache=True, project=f'ChickenCounting', name=f"{model_type}_", plots=True, pretrained=False)

        # best = f"./ChickenCounting/{model_type}/weights/best.pt"
        # if exists(best):
        #     model = YOLO(best)
        #     model.val(data='./data/data.yaml', plots=True, save_json=True)
        # else:
        #     raise Exception
        
if __name__ == "__main__":
    main()