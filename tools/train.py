# from os.path import exists

import argparse
from ultralytics import YOLO

# pretrain = True

params={
    'data': './data/data.yaml',
    'epochs': 350, 
    'batch': 0.85,
    'cache': True, 
    'project': f'ChickenCounting', 
    'plots': True, 
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

def main(l1, l2, l3):
    print(f"Going to train {l1}")
    for model_type in l1:
        print(f"Training {model_type}:")
        model = YOLO(model_type+".yaml")
        params['name'] = f"{model_type}_"
        params['pretrained'] = False
        model.train(data=params['data'],
                    epochs=params['epochs'],
                    batch=params['batch'],
                    cache=params['cache'],
                    project=params['project'],
                    name=params['name'],
                    pretrained=params['pretrained'],
                    plots=params['plots'],
                    optimizer=params['optimizer'],
                    lr0=params['lr0'],
                    lrf=params['lrf'],
                    momentum=params['momentum'],
                    weight_decay=params['weight_decay'],
                    warmup_epochs=params['warmup_epochs'],
                    warmup_momentum=params['warmup_momentum'],
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
                    copy_paste=params['copy_paste'],)
    
    #no tune
    print(f"Going to train in no tuned params {l2}")
    for model_type in l2:
        print(f"Training {model_type}:")
        model = YOLO(model_type+".yaml")
        params['name'] = f"{model_type}_nt_"
        params['pretrained'] = False
        model.train(data=params['data'],
                    epochs=params['epochs'],
                    batch=params['batch'],
                    cache=params['cache'],
                    project=params['project'],
                    name=params['name'],
                    pretrained=params['pretrained'],
                    plots=params['plots'],)
    
    #for pretrianed
    print(f"Going to train under coco pretrained {l3}")
    for model_type in l3:
        print(f"Training {model_type}:")
        model = YOLO(model_type+".yaml")
        model.load(f"ChickenCounting/{model_type}_100e_coco/weights/last.pt")
        params['name'] = f"{model_type}_cocopt_"
        params['pretrained'] = True
        model.train(data=params['data'],
                    epochs=params['epochs'],
                    batch=params['batch'],
                    cache=params['cache'],
                    project=params['project'],
                    name=params['name'],
                    pretrained=params['pretrained'],
                    plots=params['plots'],
                    optimizer=params['optimizer'],
                    lr0=params['lr0'],
                    lrf=params['lrf'],
                    momentum=params['momentum'],
                    weight_decay=params['weight_decay'],
                    warmup_epochs=params['warmup_epochs'],
                    warmup_momentum=params['warmup_momentum'],
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
                    copy_paste=params['copy_paste'],)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="根据布尔参数将对象分组")

    # 接收对象及其布尔参数
    parser.add_argument(
        "objects",
        nargs="+",
        help="输入对象和布尔参数，例如：A -t B -p C D -t -p E"
    )

    args = parser.parse_args()
    raw_objects = args.objects

    # 分组容器
    group_t = []  # 仅有 -t
    group_p = []  # 仅有 -p
    group_tp = [] # 同时有 -t 和 -p
    group_none = []  # 无 -t 和 -p 的对象

    # 当前上下文
    current_object = None
    current_flags = {"-t": False, "-p": False}

    # 解析输入
    for item in raw_objects:
        if not item.startswith("-"):  # 是对象
            if current_object:  # 将之前的对象分类到合适的组
                if current_flags["-t"] and current_flags["-p"]:
                    group_tp.append(current_object)
                    print(f"WARNING ⚠️: The Model {current_object} would not be trained!")
                elif current_flags["-t"]:
                    group_t.append(current_object)
                elif current_flags["-p"]:
                    group_p.append(current_object)
                else:
                    group_none.append(current_object)

            # 新对象
            current_object = item
            current_flags = {"-t": False, "-p": False}
        else:  # 是参数
            if item in current_flags:
                current_flags[item] = True

    # 处理最后一个对象
    if current_object:
        if current_flags["-t"] and current_flags["-p"]:
            group_tp.append(current_object)
        elif current_flags["-t"]:
            group_t.append(current_object)
        elif current_flags["-p"]:
            group_p.append(current_object)
        else:
            group_none.append(current_object)
            
    main(l1=group_none, l2=group_t, l3=group_p)