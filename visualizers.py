from src.image_preprocess import Image
import cv2
from src.object_detector import ImageObject2dList
from src.utils import Helper


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def show_image(image: Image, print_frame_id_on_image: bool = False, wait_time_msec: int = 50):

        if not isinstance(image, Image):
            raise TypeError("Input image must of type Image() only")
        
        if print_frame_id_on_image:

            label_font_type = cv2.FONT_HERSHEY_SIMPLEX
            x1 = 20
            y1 = 30

            label = f'frame id = {image.frame_id}'
        
            # Get label size
            label_size, _ = cv2.getTextSize(label, label_font_type, 0.5, 2)

            # Draw filled rectangle for label background
            cv2.rectangle(image.data, (x1, y1 - (label_size[1]+10)), (x1 + label_size[0]+50, y1), (0,0,0), cv2.FILLED)
            
            # Draw label text
            cv2.putText(image.data, label, (x1, y1 - 5), label_font_type, 0.7, (255,255,255), 2)
        
        cv2.imshow("image", image.data)
        cv2.waitKey(wait_time_msec)

    @staticmethod
    def draw_objects_on_image(image: Image, objects: ImageObject2dList, print_class_score = False, wait_time_msec = 50):

        # validate
        if not isinstance(image, Image):
            raise TypeError("Input image must of type Image() only")
        
        if not isinstance(objects, ImageObject2dList):
            raise TypeError("Input image must of type ImageObject2dList only")
        
        img = image.data.copy()

        if objects.total_objects > 0:
            for box in objects.image_objects_2d:
                x1, y1, x2, y2 = Helper.xywh_2_xyxy(box.x_min, box.y_min, box.width, box.height)

                cv2.rectangle(img, (x1,y1), (x2,y2),(0,0,255), 1)
            
                if print_class_score:
                    # label text color
                    label_text_color = (0,0,0) # black
                
                    # # label text font type
                    label_font_type = cv2.FONT_HERSHEY_SIMPLEX
                    
                    label = f'{box.object_category.name}: {int(box.confidence)} %'
        
                    # Get label size
                    label_size, _ = cv2.getTextSize(label, label_font_type, 0.5, 2)
                    
                    # # Draw filled rectangle for label background
                    cv2.rectangle(img, (x1, y1 - (label_size[1]+7)), (x1 + label_size[0], y1), (0,0,255), cv2.FILLED)
                    
                    # Draw label text
                    cv2.putText(img, label, (x1, y1 - 5), label_font_type, 0.5, label_text_color, 1)
            
        
        cv2.imshow("image_with_objects", img)
        cv2.waitKey(wait_time_msec)


    

