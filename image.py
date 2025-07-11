from src.utils import TimeStamp
import numpy as np

class Image:
    def __init__(self,
                 frame_id: int,
                 time_stamp: TimeStamp,
                 data: np.ndarray):
        
        self.frame_id = frame_id
        self.time_stamp = time_stamp
        self.data = data

        self._width = None
        self._height = None
        self._channels = None

    @property # getter
    def frame_id(self):
        return self._frame_id
    
    @frame_id.setter # setter
    def frame_id(self, value):
        if not isinstance(value, int):
            raise TypeError("Frame ID of the image must be int only")
        if not value > 0:
            raise ValueError("Frame ID must pe positive value > 0")
        self._frame_id = value

    # getter 
    @property
    def time_stamp(self):
        return self._time_stamp
    
    @time_stamp.setter
    def time_stamp(self, value):
        # validate timestamp
        if not isinstance(value, TimeStamp):
            raise TypeError("Invalid timestamp given for the image. Please check the object Timestamp for correct format")
        self._time_stamp = value

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
            raise TypeError("Image data is invalid. It must be numpy 2D array with one or more channels")
        self._data = value
        
        # set other properties to None when new image is set
        self._width = None
        self._height = None
        self._channels = None
    
    
    # only getter
    @property
    def width(self):
        if self._width is None:
            _, self._width, _ = self._data.shape
        return self._width
    
    # getter
    @property
    def height(self):
        if self._height is None:
            self._height, _, _ = self._data.shape
        return self._height

    # only getter
    @property
    def channels(self):
        if self._channels is None:
            _, _, self._channels = self._data.shape
        return self._channels
    
    def __repr__(self) -> str:
        out = "----Image-----\n" \
            + "frame id: " + str(self.frame_id) + "\n" \
            + "size: [" + str(self.height) + "," + str(self.width) + "," + str(self.channels) + "]\n" \
            + "timestamp: " + str(self.time_stamp) \
            + "values: " + str(self.data) + "\n" 
        return out