from pathlib import Path
from src.image_preprocess import ImageReader
from src.utils import CameraPosition
import cv2

camera_scene_01_path = Path(r"data/scene_01/camera_front")

all_camera_images_paths = sorted(camera_scene_01_path.glob("*.png"))

print(all_camera_images_paths)
print(len(all_camera_images_paths))

# init
camera_front_image_reader = ImageReader(camera_position=CameraPosition.CAMERA_FRONT)

for index, image_path in enumerate(all_camera_images_paths):
    img = camera_front_image_reader.read_image_from_disk(image_file_path=image_path)
    cv2.imshow("img", img.data)
    cv2.waitKey(50)
