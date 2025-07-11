from enum import Enum
from src.object_detector import (ImageObject2dList, FasterRCNN, Vgg16SSD, 
                                    YOLOv5ModelSize, YOLOv5, YOLOv8ModelSize, YOLOv8)


class ObjectDetectorType(Enum):
    Faster_RCNN_ResNet50 = 1
    Vgg16_SSD = 2
    YOLOv5_NANO = 3
    YOLOv5_SMALL = 4
    YOLOv5_MEDIUM = 5
    YOLOv5_LARGE = 6
    YOLOv5_EXTRA_LARGE = 7
    YOLOv8_NANO = 8
    YOLOv8_SMALL = 9
    YOLOv8_MEDIUM = 10
    YOLOv8_LARGE = 11
    YOLOv8_EXTRA_LARGE = 12


class ObjectDetector:
    def __init__(self, object_detector_type: ObjectDetectorType, threshold : float = 0.7):

        # validate detector type
        if not isinstance(object_detector_type, ObjectDetectorType):
            raise TypeError("Input object detector must be of type ObjectDetectorType Enum")
        
        # validate threshold value
        if not (threshold > 0 and threshold < 1):
            raise ValueError("Threshold value for the detector must be between 0 and 1. Recommeded value is 0.7.")

        self.threshold = threshold
        self.object_detector_type = object_detector_type

        # assign correct object detection as per given type
        if self.object_detector_type == ObjectDetectorType.Faster_RCNN_ResNet50:
            self.detector = FasterRCNN(threshold= self.threshold)

        # assign correct object detection as per given type
        if self.object_detector_type == ObjectDetectorType.Vgg16_SSD:
            self.detector = Vgg16SSD(threshold= self.threshold)
        
        # YOLOv5 object detectors
        if self.object_detector_type == ObjectDetectorType.YOLOv5_NANO:
            self.detector = YOLOv5(model_size=YOLOv5ModelSize.NANO, threshold= self.threshold)
        
        if self.object_detector_type == ObjectDetectorType.YOLOv5_SMALL:
            self.detector = YOLOv5(model_size=YOLOv5ModelSize.SMALL, threshold= self.threshold)

        if self.object_detector_type == ObjectDetectorType.YOLOv5_MEDIUM:
            self.detector = YOLOv5(model_size=YOLOv5ModelSize.MEDIUM, threshold= self.threshold)

        if self.object_detector_type == ObjectDetectorType.YOLOv5_LARGE:
            self.detector = YOLOv5(model_size=YOLOv5ModelSize.LARGE, threshold= self.threshold)

        if self.object_detector_type == ObjectDetectorType.YOLOv5_EXTRA_LARGE:
            self.detector = YOLOv5(model_size=YOLOv5ModelSize.EXTRA_LARGE, threshold= self.threshold)

        # YOLOv8 object detectors
        if self.object_detector_type == ObjectDetectorType.YOLOv8_NANO:
            self.detector = YOLOv8(model_size=YOLOv8ModelSize.NANO, threshold= self.threshold)
        
        if self.object_detector_type == ObjectDetectorType.YOLOv8_SMALL:
            self.detector = YOLOv8(model_size=YOLOv8ModelSize.SMALL, threshold= self.threshold)

        if self.object_detector_type == ObjectDetectorType.YOLOv8_MEDIUM:
            self.detector = YOLOv8(model_size=YOLOv8ModelSize.MEDIUM, threshold= self.threshold)

        if self.object_detector_type == ObjectDetectorType.YOLOv8_LARGE:
            self.detector = YOLOv8(model_size=YOLOv8ModelSize.LARGE, threshold= self.threshold)

        if self.object_detector_type == ObjectDetectorType.YOLOv8_EXTRA_LARGE:
            self.detector = YOLOv8(model_size=YOLOv8ModelSize.EXTRA_LARGE, threshold= self.threshold)


    def detect_objects(self, image):

        # apply object detector
        self.detector.apply_object_detector(image)

        # get the output
        objects = self.detector.get_object_list()

        return ImageObject2dList(time_stamp=image.time_stamp, frame_id=image.frame_id, 
                                 image_objects_2d=objects)
    
          
