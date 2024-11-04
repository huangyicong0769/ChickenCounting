import os
import json
import math
import cv2
import PIL.Image
from collections import OrderedDict
from labelme import utils

class Labelme2YOLO(object):
    def __init__(self, json_dir, to_seg=False):
        self._json_dir = json_dir
        self._label_id_map = self._get_label_id_map(self._json_dir)
        self._to_seg = to_seg
        self._save_path = os.path.join(self._json_dir, 'labels/')

        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

    def _get_label_id_map(self, json_dir):
        label_set = set()
        for file_name in os.listdir(json_dir):
            if file_name.endswith('json'):
                json_path = os.path.join(json_dir, file_name)
                data = json.load(open(json_path))
                for shape in data['shapes']:
                    label_set.add(shape['label'])
        return OrderedDict([(label, label_id) for label_id, label in enumerate(label_set)])

    def convert(self):
        json_names = [file_name for file_name in os.listdir(self._json_dir) 
                      if file_name.endswith('.json')]
        for json_name in json_names:
            json_path = os.path.join(self._json_dir, json_name)
            json_data = json.load(open(json_path))
            print(f'Converting {json_name} ...')

            img_path = self._save_yolo_image(json_data, json_name)
            yolo_obj_list = self._get_yolo_object_list(json_data, img_path)
            self._save_yolo_label(json_name, yolo_obj_list)

    def _get_yolo_object_list(self, json_data, img_path):
        yolo_obj_list = []
        img_h, img_w, _ = cv2.imread(img_path).shape
        for shape in json_data['shapes']:
            if shape['shape_type'] == 'circle':
                yolo_obj = self._get_circle_shape_yolo_object(shape, img_h, img_w)
            else:
                yolo_obj = self._get_other_shape_yolo_object(shape, img_h, img_w)
            yolo_obj_list.append(yolo_obj)
        return yolo_obj_list

    def _get_circle_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]
        obj_center_x, obj_center_y = shape['points'][0]
        radius = math.sqrt((obj_center_x - shape['points'][1][0]) ** 2 +
                           (obj_center_y - shape['points'][1][1]) ** 2)

        if self._to_seg:
            retval = [label_id]
            # Handle segmentation if needed (not implemented here)
            return retval

        obj_w = 2 * radius
        obj_h = 2 * radius
        yolo_center_x = round(float(obj_center_x / img_w), 6)
        yolo_center_y = round(float(obj_center_y / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        label_id = self._label_id_map[shape['label']]
        obj_x_min, obj_w, obj_y_min, obj_h = self._get_object_desc(shape['points'])

        yolo_center_x = round(float((obj_x_min + obj_w / 2.0) / img_w), 6)
        yolo_center_y = round(float((obj_y_min + obj_h / 2.0) / img_h), 6)
        yolo_w = round(float(obj_w / img_w), 6)
        yolo_h = round(float(obj_h / img_h), 6)

        return label_id, yolo_center_x, yolo_center_y, yolo_w, yolo_h

    def _get_object_desc(self, obj_port_list):
        x_lists = [port[0] for port in obj_port_list]
        y_lists = [port[1] for port in obj_port_list]
        return min(x_lists), max(x_lists) - min(x_lists), min(y_lists), max(y_lists) - min(y_lists)

    def _save_yolo_label(self, json_name, yolo_obj_list):
        txt_path = os.path.join(self._save_path, json_name.replace('.json', '.txt'))
        with open(txt_path, 'w+') as f:
            for yolo_obj in yolo_obj_list:
                f.write(' '.join(map(str, yolo_obj)) + '\n')

    def _save_yolo_image(self, json_data, json_name):
        img_name = json_name.replace('.json', '.jpg')
        img_path = os.path.join(self._json_dir, img_name)

        if not os.path.exists(img_path):
            img = utils.img_b64_to_arr(json_data['imageData'])
            PIL.Image.fromarray(img).save(img_path)

        return img_path

if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir', type=str, help='Path of the labelme json files.')
    parser.add_argument('--seg', action='store_true', help='Convert to YOLOv5 v7.0 segmentation dataset')
    args = parser.parse_args(sys.argv[1:])

    convertor = Labelme2YOLO(args.json_dir, to_seg=args.seg)
    convertor.convert()