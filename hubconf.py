dependencies = ['torch', 'yaml']
import os

import torch

from models.yolo import Model
from utils.general import set_logging
from utils.google_utils import attempt_download

set_logging()


def create(name, pretrained, channels, classes):
    """Creates a specified YOLOv5 model
    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
    Returns:
        pytorch model
    """
    config = os.path.join(os.path.dirname(__file__), 'models', f'{name}.yaml')  # model.yaml path
    try:
        model = Model(config, channels, classes)
        if pretrained:
            fname = os.path.join(os.path.dirname(__file__), 'models', f'{name}.pt'  # checkpoint filename
            ckpt = torch.load(fname, map_location=torch.device('cpu'))  # load
            state_dict = ckpt['model'].float().state_dict()  # to FP32
            state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
            model.load_state_dict(state_dict, strict=False)  # load
            if len(ckpt['model'].names) == classes:
                model.names = ckpt['model'].names  # set class names attribute
            # model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS
        return model

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache maybe be out of date, try force_reload=True. See %s for help.' % help_url
        raise Exception(s) from e


def svhn(pretrained=True, channels=3, classes=11):
    """SVHN model based on YOLOv5-small model from https://github.com/ultralytics/yolov5
    Arguments:
        pretrained (bool): load pretrained weights into the model, default=True
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=11
    Returns:
        pytorch model
    """
    return create('yolov5s-svhn', pretrained, channels, classes)