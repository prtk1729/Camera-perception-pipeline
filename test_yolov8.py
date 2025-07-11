from pathlib import Path
from src.image_preprocess import ImageReader
from src.utils import CameraPosition
from src.output_interfaces import Visualizer, FileType, FileWriter
from src.object_detector import ObjectDetector, ObjectDetectorType


camera_scene_01_path = Path(r"data/scene_02/camera_front")

all_camera_images_paths = sorted(camera_scene_01_path.glob("*.png"))

print(all_camera_images_paths)
print(len(all_camera_images_paths))

# init
camera_front_image_reader = ImageReader(camera_position=CameraPosition.CAMERA_FRONT)
object_detector = ObjectDetector(object_detector_type= ObjectDetectorType.YOLOv8_EXTRA_LARGE, threshold=0.7)
vis = Visualizer()
#file_writer = FileWriter(file_type=FileType.JSON, folder_path_to_store="out")


for index, image_path in enumerate(all_camera_images_paths):

    # reading the image from disk
    img = camera_front_image_reader.read_image_from_disk(image_file_path=image_path)

    # detect objects from one image
    objects = object_detector.detect_objects(img)

    # 
    print(objects)

    # visualizing
    #vis.show_image(image=img, print_frame_id_on_image=True, wait_time_msec=100)

    vis.draw_objects_on_image(img, objects,True, 50)

    #file_writer.write_objects_to_file(file_name=image_path.stem, object_list=objects)



    
   




