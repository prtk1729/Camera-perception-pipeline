from enum import Enum
from numbers import Real
from src.utils import TimeStamp
from typing import List


class ObjectCategory(Enum):
    PERSON = 1
    BICYCLE = 2
    MOTORBIKE = 3
    CAR = 4
    BUS = 5
    TRUCK = 6
    UNKNOWN = 7


class ImageObject2d:
    def __init__(self,
                 x_min: int = 0,
                 y_min: int = 0,
                 width: int = 0,
                 height: int = 0,
                 object_category: ObjectCategory = ObjectCategory.UNKNOWN,
                 confidence: float = 0.0):
        
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height
        self.object_category = object_category
        self.confidence = confidence

    @property # getter
    def x_min(self):
        return self._x_min
    
    @x_min.setter # setter
    def x_min(self, value):
        if not (value >= 0 and isinstance(value, int)):
            raise ValueError((f'Value of x_min must be integer pixel in {__class__.__name__} and >=0'))
        self._x_min = value

    @property # getter
    def y_min(self):
        return self._y_min
    
    @y_min.setter # setter
    def y_min(self, value):
        if not (value >= 0 and isinstance(value, int)):
            raise ValueError((f'Value of y_min must be integer pixel in {__class__.__name__} and >=0'))
        self._y_min = value

    @property # getter
    def width(self):
        return self._width
    
    @width.setter # setter
    def width(self, value):
        if not (value > 0 and isinstance(value, int)):
            raise ValueError((f'Value of width must be integer pixel in {__class__.__name__} and >0'))
        self._width = value
    
    @property # getter
    def height(self):
        return self._height
    
    @height.setter # setter
    def height(self, value):
        if not (value > 0 and isinstance(value, int)):
            raise ValueError((f'Value of height must be integer pixel in {__class__.__name__} and >0'))
        self._height = value

    @property # getter
    def confidence(self):
        return self._confidence
    
    @confidence.setter # setter
    def confidence(self, value):
        if isinstance(value, Real):
            if not (value >=0 and value <=100):
                raise ValueError(f'Value of object confidence must be between [0,100] in {__class__.__name__}. But confidence = {value} is given.')
            else:
                self._confidence = int(value)
        else:
            raise TypeError(f'Value of object confidence must be numeric in {__class__.__name__}. But confidence = {value} of type {type(value)} is given.')
    
    @property # getter
    def object_category(self):
        return self._object_category
    
    @object_category.setter # setter
    def object_category(self, value):
        if not isinstance(value, ObjectCategory):
            raise TypeError(f'Object category must of enum type  {ObjectCategory.__name__} in {__class__.__name__}. But category = {value} of type {type(value)} is given.')
        self._object_category = value
    

    def __repr__(self):
        out = f'{__class__.__name__} - [x_min = {self.x_min}, y_min = {self.y_min}, width = {self.width}, height = {self.height}, object category = {self.object_category.name}, confidence = {self.confidence}]'
        return out
            

class ImageObject2dList:
    def __init__(self,
                 time_stamp: TimeStamp,
                 frame_id : int,
                 image_objects_2d: List[ImageObject2d] = []):
        
        self.time_stamp = time_stamp
        self.frame_id = frame_id
        self.image_objects_2d = image_objects_2d
        self._total_objects = 0

    
    @property # getter
    def frame_id(self):
        return self._frame_id
     
    @frame_id.setter # setter
    def frame_id(self, value):
        if isinstance(value, int):
            if value < 0:
                raise ValueError(f'Value of frame id in {__class__.__name__} must be a positive integer. But frame id = {value} of type {type(value)} is given.') 
            else:
                self._frame_id = value
        else: 
            raise TypeError(f'Value of frame id must be of type int in {__class__.__name__}. But frame id = {value} of type {type(value)} is given.')
    
    @property # getter
    def time_stamp(self):
        return self._time_stamp
    
    @time_stamp.setter
    def time_stamp(self, value):
        if not isinstance(value, TimeStamp):
            raise TypeError(f'Value of time stamp must be of type TimeStamp in {__class__.__name__}. But time stamp = {value} of type {type(value)} is given.')
        self._time_stamp = value
    
    @property # getter
    def image_objects_2d(self):
        return self._image_objects_2d
    
    @image_objects_2d.setter
    def image_objects_2d(self, value):
        if isinstance(value, List):
            if len(value) == 0: # list is empty
                self._image_objects_2d = value
            else:
                for obj in value:
                    if not isinstance(obj, ImageObject2d):
                        raise TypeError("List must contain all objects of type ImageObject2d only")
                self._image_objects_2d = value
        else:
            raise TypeError("Value of image objects 2d must be given in form of List only")
    
    @property # getter
    def total_objects(self):
        return len(self.image_objects_2d)
    
    def __repr__(self):
        str_object_list = ""
        for index, obj in enumerate(self.image_objects_2d):
            str_object_list += str(index+1) + ". " + str(obj) +'\n'
        
        return(
            f'-------------------------------------\n'
            f'{__class__.__name__}: epoch time = {self.time_stamp}, frame id = {self.frame_id}, total objects = {self.total_objects}\n'
            f'{str_object_list}'
            f'-------------------------------------\n' 
        )

        

