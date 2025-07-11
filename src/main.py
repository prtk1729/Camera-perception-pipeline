from src.top_level_api import CameraPerception
from src.utils import CameraPosition
from src.object_detector import ObjectDetectorType
from src.output_interfaces import FileType


if __name__ == "__main__":

    my_camera_perception = CameraPerception(folder_path_of_image_files=r"data/scene_01/camera_front",
                                            camera_position=CameraPosition.CAMERA_FRONT,
                                            object_detector_type= ObjectDetectorType.YOLOv5_MEDIUM,
                                            object_detector_threshold= 0.7,
                                            file_type=FileType.JSON,
                                            output_folder_to_store="out",
                                            output_at_console=True,
                                            visualize_raw_image=True,
                                            visualize_objects_on_image=True,
                                            write_object_list_to_file=True)
    
    my_camera_perception.run()
