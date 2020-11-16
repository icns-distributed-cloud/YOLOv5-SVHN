# YOLOv5-SVHN
CNN Model based on [YOLOv5](https://github.com/ultralytics/yolov5) for [SVHN dataset](http://ufldl.stanford.edu/housenumbers) (format 2).
Especially, used YOLOv5s model.

## Usage
<pre>
<code>
# Download and Reshape the model
import torch
model = torch.hub.load('icns-distributed-cloud/yolov5-svhn', 'svhn').fuse().eval()
model = model.autoshape()

# Prediction
prediction = model(img, size=640)
for x1, y1, x2, y2, conf, clas in pred:
    print('box: ({}, {}), ({}, {})'.format(x1, y1, x2, y2))
    print('confidence : {}'.format(conf))
    print('class: {}'.format(int(clas)))
</code>
</pre>

## References
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [SVHN Datasets](http://ufldl.stanford.edu/housenumbers)
