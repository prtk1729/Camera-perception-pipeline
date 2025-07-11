from enum import Enum
from src.object_detector import ObjectDetectorAbstract, ImageObject2d, ObjectCategory
import torch
from src.utils import Helper


class YOLOv5ModelSize(Enum):
    NANO = 1
    SMALL = 2
    MEDIUM = 3
    LARGE = 4
    EXTRA_LARGE = 5


class YOLOv5(ObjectDetectorAbstract):

    def __init__(self, 
                 model_size: YOLOv5ModelSize.MEDIUM,
                 threshold : float = 0.7,
                 nms_suppression_threshold: float= 0.6):
        
        super().__init__()

        # validate model size
        if not isinstance(model_size, YOLOv5ModelSize):
            raise TypeError("Invalid model size given to YOLOv5")
        self.model_size = model_size

        # validate thrshold value
        if threshold <=0 and threshold > 1:
            raise ValueError("Threshold value must be >0 and <1")
        self.threshold = threshold

        # validate nms_suppression_threshold value
        if nms_suppression_threshold <=0 and nms_suppression_threshold > 1:
            raise ValueError("nms_suppression_threshold value must be >0 and <1")
        self.nms_suppression_threshold = nms_suppression_threshold

        #
        self.model_names = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l','yolov5x']

        # print
        print(f"yolov5 {self.model_names[model_size.value-1]}")

        # 
        self.model = torch.hub.load('ultralytics/yolov5', self.model_names[model_size.value-1], pretrained = True,
                                    force_reload = True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # labels defined as per COCO dataset 
        self.labels = ['__background__', 'person', 'bicycle', 'car', 'motorbike', 'airplane', 'bus',
                                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
                                        'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                                        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                                        'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
                                        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                                        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                                        'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
                                        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                                        'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
                                        'teddy bear', 'hair drier', 'toothbrush']
        
        self.model.conf = self.threshold
        self.model.iou = self.nms_suppression_threshold

    def apply_object_detector(self, image):
        
        # prediction / inference
        predictions = self.model(image.data)

        # extract bboxes, transfer to cpu and convert tensors to numpy
        boxes = predictions.xyxy[0]
        boxes = boxes.cpu().numpy()

        object_list = []

        # box - [x1, y1, x2, y2, score, class_id]

        for box in boxes:
            
            class_label_id = int(box[5])
            
            if self.labels[class_label_id + 1] in self.valid_categories:

                # extract bounding box
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                x, y, w,h = Helper.xyxy_2_xywh(x1,y1,x2,y2)

                object_confidence = int(box[4]*100)

                # object category
                obj_category =  self.labels[class_label_id + 1]

                obj = ImageObject2d(x_min = x, 
                                    y_min = y,
                                    width=w,
                                    height=h,
                                    object_category= ObjectCategory[obj_category.upper()],
                                    confidence=object_confidence)

                object_list.append(obj)
        

        self.object_list = object_list
    
    def get_object_list(self):
        return self.object_list
        







