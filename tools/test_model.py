# from os.path import exists

from ultralytics import YOLO
import argparse

def main(args):
        model_type = args.model
        model = YOLO(model_type+".yaml")
        model.train(data='./data/data.yaml', epochs=25, batch=32, cache=True, pretrained=False, save=False, plots=False)
        model.val(data='./data/data.yaml')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test model")
    parser.add_argument("model", type=str, help="model config")
    main(parser.parse_args())