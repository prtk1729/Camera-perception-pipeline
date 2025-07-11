from enum import Enum
from src.object_detector import ObjectDetectorAbstract, ImageObject2d, ObjectCategory
from ultralytics import YOLO
import torch
import cv2 as cv
from src.utils import Helper

class YOLOv8ModelSize(Enum):
    NANO = 1
    SMALL = 2
    MEDIUM = 3
    LARGE = 4
    EXTRA_LARGE = 5


class YOLOv8(ObjectDetectorAbstract):
    def __init__(self, model_size: YOLOv8ModelSize = YOLOv8ModelSize.MEDIUM,
                 threshold: float = 0.7,
                 nms_suppression_threshold : float = 0.6
                 ):
        # 
        super().__init__()

        # validate model size
        if not isinstance(model_size, YOLOv8ModelSize):
            raise TypeError("Invalid model size given to YOLOv8")
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
        self.model_names = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt','yolov8x.pt']

        #
        self.model = YOLO(self.model_names[model_size.value -1])

        # print
        print(f"from yolov8 object detection - model {self.model_names[model_size.value -1]} is used")

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
        

    def apply_object_detector(self, image):  
        
        # resize 
        image_resize = cv.resize(image.data, (640,640), interpolation = cv.INTER_AREA)

        # inference / prediction
        predictions = self.model.predict(image_resize, conf = self.threshold, iou = self.nms_suppression_threshold)

        # transfer the predictions to cpu and then convert tensors to numpy
        predictions = predictions[0].cpu().numpy()

        # boxes with class and score
        boxes = predictions.boxes

        object_list = []

        for box in boxes:
            if self.labels[int(box.cls + 1)] in self.valid_categories:

                # extract bounding box values

                x1 = int(box.xyxy[0][0])
                y1 = int(box.xyxy[0][1])
                x2 = int(box.xyxy[0][2])
                y2 = int(box.xyxy[0][3])

                # resize the bounding box to image size
                x1 = int((x1*image.width)/640)
                y1 = int((y1*image.height)/640)
                x2 = int((x2*image.width)/640)
                y2 = int((y2*image.height)/640)

                # apply image size rstrictions to th output bounding boxes
                if x1 < 0: x1 = 0
                if x1 > image.width: x1 = image.width

                if x2 < 0: x2 = 0
                if x2 > image.width: x2 = image.width

                if y1 < 0: y1 = 0
                if y1 > image.height: y1 = image.height

                if y2 < 0: y2 = 0
                if y2 > image.height: y2 = image.height

                x, y, w, h  = Helper.xyxy_2_xywh(x1,y1,x2,y2)

                # object category
                object_category = self.labels[int(box.cls + 1)]

                # object score
                obj_confidence = int(box.conf * 100)

                obj = ImageObject2d(x_min = x, 
                                    y_min = y,
                                    width=w,
                                    height=h,
                                    object_category= ObjectCategory[object_category.upper()],
                                    confidence=obj_confidence)

                object_list.append(obj)
        

        self.object_list = object_list

    def get_object_list(self):
        return self.object_list 