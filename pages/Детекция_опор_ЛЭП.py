from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import settings
import yolo_helper
import cv2

def gestures_demo():
    st.set_page_config(
        page_title="Распознавание опор ЛЭП",
        page_icon="🗼",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Распознавание опор ЛЭП")
    st.sidebar.header("Настройки")

    path_to_json_config = 'yolo_config.json'
    config_loader = yolo_helper.ConfigLoader(path_to_json_config)
    config = config_loader.get_config()

    translator = config["to_russian_utility_towers"]
    confidence = config["utility_tower_detector_conf"]
    weights = config["utility_tower_detector_path"]
    try:
        yolo_v8_class_obj = yolo_helper.YOLOv8Model()
        model = yolo_v8_class_obj.load_model(weights)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {config['utility_tower_detector_path']}")
        st.error(ex)

    st.sidebar.header("Настройки")
    source_radio = st.sidebar.radio("Выберите источник: ", settings.SOURCES_LIST_UTILITY)

    if source_radio == settings.IMAGE:
        image_file = st.file_uploader("Загрузите ваше изображение", type=['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'PNG'])

        if not image_file:
            return None
        
        original_image = Image.open(image_file)
        original_image = ImageOps.exif_transpose(original_image)
        original_image_np = np.asarray(original_image)
        original_image_np = original_image_np.copy()

        results = yolo_v8_class_obj.detect_objects(frame=original_image_np,
                                                model=model,
                                                current_model_conf=confidence,
                                                image_size=1024,
                                                image_displayer=yolo_helper.ImageDisplayer(),
                                                labels_translator=translator)
        # st.text(f"{results}")
        st.text("Результаты распознавания")
        col1, col2 = st.columns(2)
        col1.image(original_image, caption = "Original image")
        col2.image(original_image_np, caption = "Detection results")
        # st.image(cutout_images, clamp=True)
    else:
        st.error("Please select a valid source type!")


gestures_demo()

# show_code(plotting_demo)
