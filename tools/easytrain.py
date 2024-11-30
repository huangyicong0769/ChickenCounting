# from os.path import exists

from ultralytics import YOLO

# pretrain = True

meta_params={
    'data': './data/data.yaml',
    'epochs': 350, 
    'batch': 0.85,
    'cache': True, 
    'project': f'ChickenCounting', 
    'plots': True, 
    'patience': 0,
    'iou_type': 'CIoU',
    'close_mosaic': 20,
    'pretrained': False
}

hyp_params={
    'optimizer': 'AdamW',
    'lr0': 0.00732,
    'lrf': 0.01298,
    'momentum': 0.83868,
    'weight_decay': 0.0006,
    'warmup_epochs': 1.83822,
    'warmup_momentum': 0.59035,
    'box': 6.88413,
    'cls': 0.84057,
    'dfl': 1.46567,
    'hsv_h': 0.01287,
    'hsv_s': 0.89415,
    'hsv_v': 0.2089,
    'degrees': 0.0,
    'translate': 0.09686,
    'scale': 0.3474,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.36604,
    'bgr': 0.0,
    'mosaic': 0.82585,
    'mixup': 0.0,
    'copy_paste': 0.0,
}

hyp_params_SGD={
    'optimizer': 'SGD',
    'lr0': 0.00745,
    'lrf': 0.00758,
    'momentum': 0.96777,
    'weight_decay': 0.00085,
    'warmup_epochs': 1.84925,
    'warmup_momentum': 0.93664,
    'box': 4.7666,
    'cls': 0.45789,
    'dfl': 1.19228,
    'hsv_h': 0.01495,
    'hsv_s': 0.67288,
    'hsv_v': 0.32582,
    'degrees': 0.0,
    'translate': 0.07308,
    'scale': 0.44555,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.39590,
    'bgr': 0.0,
    'mosaic': 0.65968,
    'mixup': 0.0,
    'copy_paste': 0.0,
}

def main():
    model_type = 'yolov8s_MCWA_AConv_C2INXB_GD'
    params = meta_params | hyp_params_SGD
  
    #1
    print(f"Training {model_type}:")
    model = YOLO(model_type+".yaml").load(r"ChickenCounting/yolov8s_MCWA_AConv_C2INXB_GD_100e_coco/weights/last.pt")
    params['name'] = f"{model_type}_SGD_cocopt_"
    params['pretrained'] = True
    model.train(data=params['data'],
                epochs=params['epochs'],
                batch=params['batch'],
                cache=params['cache'],
                project=params['project'],
                name=params['name'],
                pretrained=params['pretrained'],
                patience=params['patience'],
                plots=params['plots'],
                close_mosaic=params['close_mosaic'],
                optimizer=params['optimizer'],
                lr0=params['lr0'],
                lrf=params['lrf'],
                momentum=params['momentum'],
                weight_decay=params['weight_decay'],
                warmup_epochs=params['warmup_epochs'],
                warmup_momentum=params['warmup_momentum'],
                # kp = 1./(params['lr0']*params['momentum']),
                # ki = 0.1,
                # kd = 1.,
                box=params['box'],
                cls=params['cls'],
                dfl=params['dfl'],
                hsv_h=params['hsv_h'],
                hsv_s=params['hsv_s'],
                hsv_v=params['hsv_v'],
                degrees=params['degrees'],
                translate=params['translate'],
                scale=params['scale'],
                shear=params['shear'],
                perspective=params['perspective'],
                flipud=params['flipud'],
                fliplr=params['fliplr'],
                bgr=params['bgr'],
                mosaic=params['mosaic'],
                mixup=params['mixup'],
                copy_paste=params['copy_paste'],
                exist_ok=True)

if __name__ == "__main__":
    main()