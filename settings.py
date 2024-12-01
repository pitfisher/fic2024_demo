from pathlib import Path
import sys

file_path = Path(__file__).resolve()

root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

VIDEO_DIR = ROOT / "media/videos/"
IMAGES_DIR = ROOT / 'media/images/'

# Sources
WEBCAM = 'Веб-камера'
VIDEO = 'Видео'
IMAGE = 'Изображение'
IMAGES_DIRECTORY = 'Папка'
SOURCES_LIST_UTILITY = [IMAGE]
# Webcam
WEBCAM_ID = 0
