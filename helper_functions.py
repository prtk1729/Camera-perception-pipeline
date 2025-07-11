import numpy as np


class Helper:
    def __init__(self):
        pass

    @staticmethod
    def xyxy_2_xywh(x1: int, y1: int, x2: int, y2: int):

        x_min = int(x1)
        y_min = int(y1)

        width = int(abs(x2-x1))
        height = int(abs(y2-y1))

        return x_min, y_min, width, height
    

    @staticmethod
    def xywh_2_xyxy(x_min: int, y_min: int, width: int, height: int):

        x1 = int(x_min)
        y1 = int(y_min)
        x2 = int(x1 + width)
        y2 = int(y1 + height)

        return x1, y1, x2, y2
    

    
    
