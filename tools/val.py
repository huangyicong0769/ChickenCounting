from ultralytics import YOLO

params={
    'data': './data/data.yaml',
    'project': f'ChickenCounting', 
    'plots': True,
}

def main():
    model = YOLO(r"ChickenCounting/yolov8s_WT_MCA_AConv_C2INXB_GD_3/weights/best.pt")
    rlt = model.val(data=params['data'], project=params['project'], name='yolov8s_WT_MCA_AConv_C2INXB_GD_', plots=params['plots'])
    print(rlt.box.map)
    print(rlt.box.map50)
    print(rlt.box.map75)
    print(rlt.box.maps)

if __name__ == '__main__':
    main()