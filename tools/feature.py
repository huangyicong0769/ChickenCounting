import argparse
from ultralytics import YOLO

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class yolov8_heatmap:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        model.load_state_dict(csd, strict=False)  # load
        model.eval()
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')

        target_layers = [eval(layer)]
        method = eval(method)

        self.colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
            indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path):
        # remove dir if exist
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # init ActivationsAndGradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # get ActivationsAndResult
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # postprocess to yolo output
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf_threshold:
                break

            self.model.zero_grad()
            # get max probability for this prediction
            if self.backward_type == 'class' or self.backward_type == 'all':
                score = post_result[i].max()
                score.backward(retain_graph=True)

            if self.backward_type == 'box' or self.backward_type == 'all':
                for j in range(4):
                    score = pre_post_boxes[i, j]
                    score.backward(retain_graph=True)

            # process heatmap
            if self.backward_type == 'class':
                gradients = grads.gradients[0]
            elif self.backward_type == 'box':
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
            else:
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
                            grads.gradients[4]
            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                  gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            if (saliency_map_max - saliency_map_min) == 0:
                continue
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

            # add heatmap and box to image
            cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
            cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, :].argmax())],
                                             f'{self.model_names[int(post_result[i, :].argmax())]} '
                                             f'{float(post_result[i].max()):.2f}',
                                             cam_image)    ## 不用预测框则注释 
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/{i}.png')

def get_params(args):
    params = {
        'weight': args.weight, # 这选择想要热力可视化的模型权重路径
        'cfg': args.config, # 这里选择与训练上面模型权重相对应的.yaml文件路径
        'device': 'cuda:0', # 选择设备，其中0表示0号显卡。如果使用CPU可视化 # 'device': 'cpu'
        'method': 'XGradCAM', # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': f'model.model[{args.layer}]',   # 选择特征层
        'backward_type': 'all', # class, box, all
        'conf_threshold': 0.65, # 置信度阈值默认0.65， 可根据情况调节
        'ratio': 0.005 # 取前多少数据，默认是0.02，可根据情况调节
    }
    return params

def main(args):
    model = yolov8_heatmap(**get_params(args)) # 初始化
    model(args.file, args.output) # 第一个参数是图片的路径，第二个参数是保存路径，比如是result的话，其会创建一个名字为result的文件夹，如果result文件夹不为空，其会先清空文件夹。
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict on specific image given model")
    parser.add_argument("file", help="directory of image")
    parser.add_argument("-w", "--weight", required=True, help="directory of model weight")
    parser.add_argument("-c", "--config", required=True, help="directory of model config")
    parser.add_argument("-l", "--layer", type=int, default=-1, help="target layer")
    parser.add_argument("-o", "--output", default='./result', help="directory of result")
    parser.add_argument("-hm", "--heat_map", type=bool, default=True, help="Output heat map")
    main(args=parser.parse_args())