import os
import time
import hashlib
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from ultralytics import YOLO
from deep_translator import GoogleTranslator

from disease_info import disease_info
from pdf_report import generate_pdf
from weather import get_weather

import base64
from io import BytesIO
from gtts import gTTS


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)


# =========================================================
# LOAD MODELS
# Models will load only once
# =========================================================
@st.cache_resource
def load_models():

    cnn_model = tf.keras.models.load_model(
        "best_model.h5",
        compile=False
    )

    yolo_model = YOLO("yolov8n.pt")

    with open("class_names.txt", "r", encoding="utf-8") as file:
        class_names = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    return cnn_model, yolo_model, class_names


cnn_model, yolo, class_names = load_models()


# =========================================================
# SESSION STATE
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "processed_images" not in st.session_state:
    st.session_state.processed_images = []


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def pretty_name(text):
    """
    Convert:
    Tomato___Early_blight
    or
    Early_blight

    into readable text.
    """

    return text.replace("_", " ").strip()


def cnn_predict(image_rgb):
    """
    Predict plant disease using CNN model.
    """

    image = cv2.resize(image_rgb, (224, 224))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = cnn_model.predict(
        image,
        verbose=0
    )[0]

    index = int(np.argmax(prediction))

    confidence = float(np.max(prediction))

    if index >= len(class_names):
        raise ValueError(
            "CNN output classes and class_names.txt do not match."
        )

    class_name = class_names[index]

    if "___" in class_name:
        plant, disease = class_name.split("___", 1)
    else:
        plant = "Unknown Plant"
        disease = class_name

    return plant, disease, confidence


def coco_detect(image):
    """
    Detect obvious non-leaf objects using YOLOv8.

    NOTE:
    YOLO COCO is not a real leaf/non-leaf classifier.
    It is only being used as an additional object filter.
    """

    results = yolo(
        image,
        verbose=False
    )[0]

    if results.boxes is None:
        return None

    if len(results.boxes) == 0:
        return None

    # Find highest-confidence detected object
    confidences = results.boxes.conf.cpu().numpy()

    best_index = int(np.argmax(confidences))

    best_confidence = float(confidences[best_index])

    # Ignore weak detections
    if best_confidence < 0.50:
        return None

    box = results.boxes[best_index]

    class_id = int(
        box.cls[0].item()
    )

    label = yolo.names[class_id]

    # A plant itself should not automatically be rejected
    allowed_labels = {
        "potted plant"
    }

    if label.lower() in allowed_labels:
        return None

    coordinates = box.xyxy[0].cpu().numpy()

    return label, coordinates, best_confidence


def translate_text(text, selected_language):

    if selected_language == "English":
        return str(text)

    language_map = {
        "Hindi": "hi",
        "Bengali": "bn",
        "Punjabi": "pa",
        "Urdu": "ur",
        "Tamil": "ta"
    }

    target_language = language_map.get(
        selected_language
    )

    if target_language is None:
        return str(text)

    try:

        translated_text = GoogleTranslator(
            source="auto",
            target=target_language
        ).translate(str(text))

        return translated_text

    except Exception:

        # If translation API fails,
        # show original English text
        return str(text)
    # =========================================================
# VOICE ASSISTANT FUNCTIONS
# =========================================================

def get_voice_language_code(selected_language):

    voice_language_map = {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn",
        "Punjabi": "pa",
        "Urdu": "ur",
        "Tamil": "ta"
    }

    return voice_language_map.get(
        selected_language,
        "en"
    )


def build_voice_report(
    plant,
    disease,
    confidence,
    weather
):

    plant_name = pretty_name(plant)
    disease_name = pretty_name(disease)

    voice_text = ""

    # Plant name
    voice_text += (
        f"Plant detected is {plant_name}. "
    )

    # Disease name
    voice_text += (
        f"Detected disease is {disease_name}. "
    )

    # Confidence
    voice_text += (
        f"The prediction confidence is "
        f"{confidence * 100:.2f} percent. "
    )

    # Confidence status
    if confidence >= 0.90:

        voice_text += (
            "The artificial intelligence prediction "
            "confidence is excellent. "
        )

    elif confidence >= 0.70:

        voice_text += (
            "The artificial intelligence prediction "
            "confidence is good. "
        )

    else:

        voice_text += (
            "The prediction confidence is low. "
            "Please try another clear image of the leaf. "
        )


    # =====================================================
    # WEATHER INFORMATION
    # =====================================================
    if weather:

        temperature = weather.get(
            "temp",
            "not available"
        )

        humidity = weather.get(
            "humidity",
            "not available"
        )

        weather_condition = weather.get(
            "weather",
            "not available"
        )

        voice_text += (
            f"The current temperature is "
            f"{temperature} degrees Celsius. "
        )

        voice_text += (
            f"The current humidity is "
            f"{humidity} percent. "
        )

        voice_text += (
            f"The weather condition is "
            f"{weather_condition}. "
        )

        try:

            if float(humidity) >= 80:

                voice_text += (
                    "Warning. Humidity is high. "
                    "Plant disease risk may also be high. "
                    "Avoid over watering and use appropriate "
                    "treatment if necessary. "
                )

            else:

                voice_text += (
                    "The current weather conditions are normal. "
                )

        except (ValueError, TypeError):
            pass


    # =====================================================
    # DISEASE INFORMATION
    # =====================================================
    info = disease_info.get(
        disease
    )

    if info:

        # Description
        description = info.get(
            "description"
        )

        if description:

            voice_text += (
                f"Disease description. "
                f"{description}. "
            )


        # Symptoms
        symptoms = info.get(
            "symptoms",
            []
        )

        if symptoms:

            voice_text += (
                "The main symptoms are. "
            )

            for symptom in symptoms:

                voice_text += (
                    f"{symptom}. "
                )


        # Treatment
        treatments = info.get(
            "treatment",
            []
        )

        if treatments:

            voice_text += (
                "Recommended treatment. "
            )

            for treatment in treatments:

                voice_text += (
                    f"{treatment}. "
                )


        # Prevention
        preventions = info.get(
            "prevention",
            []
        )

        if preventions:

            voice_text += (
                "Prevention methods. "
            )

            for prevention in preventions:

                voice_text += (
                    f"{prevention}. "
                )

    return voice_text


def generate_voice(
    text,
    selected_language
):

    # Selected language me translate karega
    translated_voice_text = translate_text(
        text,
        selected_language
    )

    # Voice language code lega
    language_code = get_voice_language_code(
        selected_language
    )

    # Voice create karega
    tts = gTTS(
        text=translated_voice_text,
        lang=language_code,
        slow=False
    )

    # Audio memory me save karega
    audio_buffer = BytesIO()

    tts.write_to_fp(
        audio_buffer
    )

    audio_buffer.seek(0)

    return audio_buffer.getvalue()


def autoplay_voice(
    audio_bytes
):

    audio_base64 = base64.b64encode(
        audio_bytes
    ).decode()

    audio_html = f"""
    <audio autoplay controls style="width:100%;">
        <source
            src="data:audio/mp3;base64,{audio_base64}"
            type="audio/mp3"
        >
    </audio>
    """

    st.markdown(
        audio_html,
        unsafe_allow_html=True
    )


def show_voice_assistant(
    plant,
    disease,
    confidence,
    weather,
    selected_language
):

    st.markdown("---")

    st.subheader(
        "🔊 AI Voice Assistant"
    )

    st.info(
        f"Click the button to hear the complete "
        f"plant disease report in {selected_language}."
    )

    button_key = (
        f"voice_"
        f"{plant}_"
        f"{disease}_"
        f"{selected_language}"
    )

    if st.button(
        f"🔊 Speak Full Report in {selected_language}",
        key=button_key
    ):

        try:

            with st.spinner(
                f"🔊 Generating {selected_language} voice..."
            ):

                voice_report = build_voice_report(
                    plant=plant,
                    disease=disease,
                    confidence=confidence,
                    weather=weather
                )

                audio_data = generate_voice(
                    text=voice_report,
                    selected_language=selected_language
                )

                autoplay_voice(
                    audio_data
                )

            st.success(
                f"✅ Playing report in {selected_language}"
            )

        except Exception as error:

            st.error(
                "❌ Voice could not be generated."
            )

            st.warning(
                "Please check your internet connection."
            )

            st.exception(
                error
            )


@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(city_name):

    if not city_name.strip():
        return None

    try:
        return get_weather(
            city_name.strip()
        )

    except Exception:
        return None


def add_to_history(
    image_id,
    plant,
    disease,
    confidence
):

    # Prevent duplicate history on every Streamlit rerun
    if image_id in st.session_state.processed_images:
        return

    st.session_state.history.append(
        {
            "Date": datetime.now().strftime(
                "%d-%m-%Y"
            ),

            "Time": datetime.now().strftime(
                "%I:%M:%S %p"
            ),

            "Plant": pretty_name(plant),

            "Disease": pretty_name(disease),

            "Confidence": (
                f"{confidence * 100:.2f}%"
            )
        }
    )

    st.session_state.processed_images.append(
        image_id
    )


def show_weather_information(
    weather
):

    if not weather:
        return

    st.markdown("---")

    st.subheader(
        "🌦 Current Weather"
    )

    st.write(
        f"🌡 Temperature: "
        f"{weather.get('temp', 'N/A')} °C"
    )

    st.write(
        f"💧 Humidity: "
        f"{weather.get('humidity', 'N/A')} %"
    )

    st.write(
        f"☁ Weather: "
        f"{weather.get('weather', 'N/A')}"
    )

    humidity = weather.get(
        "humidity"
    )

    if humidity is not None:

        try:

            humidity = float(
                humidity
            )

            if humidity >= 80:

                st.warning(
                    "⚠ High humidity. "
                    "Disease risk is HIGH."
                )

                st.info(
                    "💡 Recommendation: "
                    "Avoid over-watering and "
                    "use suitable treatment "
                    "if required."
                )

            else:

                st.success(
                    "✅ Weather conditions "
                    "are normal."
                )

        except (ValueError, TypeError):
            pass


def show_disease_information(
    disease,
    selected_language
):

    # Get disease information safely
    info = disease_info.get(
        disease
    )

    if info is None:

        st.info(
            "Detailed information for this "
            "disease is not available yet."
        )

        return

    st.markdown("---")

    st.subheader(
        "📋 Disease Information"
    )

    # ---------------- Description ----------------
    st.write(
        "### 📝 Description"
    )

    st.write(
        translate_text(
            info.get(
                "description",
                "No description available."
            ),
            selected_language
        )
    )

    # ---------------- Symptoms ----------------
    st.write(
        "### 🔍 Symptoms"
    )

    symptoms = info.get(
        "symptoms",
        []
    )

    if symptoms:

        for symptom in symptoms:

            st.write(
                f"• "
                f"{translate_text(symptom, selected_language)}"
            )

    else:

        st.write(
            "No symptoms information available."
        )

    # ---------------- Treatment ----------------
    st.write(
        "### 💊 Treatment"
    )

    treatments = info.get(
        "treatment",
        []
    )

    if treatments:

        for treatment in treatments:

            st.write(
                f"• "
                f"{translate_text(treatment, selected_language)}"
            )

    else:

        st.write(
            "No treatment information available."
        )

    # ---------------- Prevention ----------------
    st.write(
        "### 🛡 Prevention"
    )

    preventions = info.get(
        "prevention",
        []
    )

    if preventions:

        for prevention in preventions:

            st.write(
                f"• "
                f"{translate_text(prevention, selected_language)}"
            )

    else:

        st.write(
            "No prevention information available."
        )


def show_confidence(
    confidence
):

    st.markdown("---")

    st.subheader(
        "🎯 Prediction Confidence"
    )

    confidence = float(
        np.clip(
            confidence,
            0.0,
            1.0
        )
    )

    st.progress(
        confidence
    )

    st.write(
        f"**{confidence * 100:.2f}%**"
    )

    if confidence >= 0.90:

        st.success(
            "🟢 Excellent Prediction"
        )

    elif confidence >= 0.70:

        st.warning(
            "🟡 Good Prediction"
        )

    else:

        st.error(
            "🔴 Low Confidence - "
            "Try another clear leaf image."
        )


def create_pdf_download(
    plant,
    disease,
    confidence,
    weather
):

    st.markdown("---")

    st.subheader(
        "📄 PDF Report"
    )

    try:

        pdf_file = generate_pdf(
            plant=pretty_name(plant),

            disease=pretty_name(disease),

            confidence=confidence * 100,

            weather=weather
        )

        if os.path.exists(pdf_file):

            with open(
                pdf_file,
                "rb"
            ) as file:

                st.download_button(
                    label="📄 Download PDF Report",

                    data=file.read(),

                    file_name=(
                        "Plant_Disease_Report.pdf"
                    ),

                    mime="application/pdf"
                )

        else:

            st.warning(
                "PDF file could not be created."
            )

    except Exception as error:

        st.warning(
            f"PDF report could not be generated: "
            f"{error}"
        )


def show_prediction_result(
    frame,
    plant,
    disease,
    confidence,
    weather,
    selected_language
):

    st.markdown("---")

    st.subheader(
        "🧠 AI Prediction Result"
    )

    plant_name = pretty_name(
        plant
    )

    disease_name = pretty_name(
        disease
    )

    st.success(
        translate_text(
            f"Plant: {plant_name}",
            selected_language
        )
    )

    st.warning(
        translate_text(
            f"Disease: {disease_name}",
            selected_language
        )
    )

    st.info(
        translate_text(
            (
                f"Confidence: "
                f"{confidence * 100:.2f}%"
            ),
            selected_language
        )
    )

    # Weather
    show_weather_information(
        weather
    )

    # Disease details
    show_disease_information(
        disease,
        selected_language
    )

    # Confidence meter
    show_confidence(
        confidence
    )

    # Image
    st.markdown("---")

    st.subheader(
        "🖼 Analyzed Image"
    )

    st.image(
        frame,
        width=700
    )

    # PDF
    create_pdf_download(
        plant,
        disease,
        confidence,
        weather
    )
    # Voice Assistant
    show_voice_assistant(
        plant=plant,
        disease=disease,
        confidence=confidence,
        weather=weather,
        selected_language=selected_language
    )

# =========================================================
# MAIN TITLE
# =========================================================
st.title(
    "🌿 Plant Disease Detection"
)

st.write(
    "Upload a plant leaf image, video, "
    "or use your camera to detect plant diseases."
)


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title(
    "🌿 LeafGuard AI"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "👨‍💻 Developer"
)

st.sidebar.write(
    "Md Asgar Ansari"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "🤖 AI Models"
)

st.sidebar.success(
    "YOLOv8"
)

st.sidebar.success(
    "TensorFlow CNN"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "✨ Features"
)

st.sidebar.write(
    "✅ Image Detection"
)

st.sidebar.write(
    "✅ Video Detection"
)

st.sidebar.write(
    "✅ Camera Detection"
)

st.sidebar.write(
    "✅ Weather Analysis"
)

st.sidebar.write(
    "✅ Multi-Language Support"
)

st.sidebar.write(
    "✅ PDF Report"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📌 Version"
)

st.sidebar.info(
    "Version 1.0"
)

st.sidebar.success(
    "🟢 System Ready"
)

st.sidebar.markdown("---")

# History container will be filled at the end
history_container = st.sidebar.container()


# =========================================================
# INPUT SETTINGS
# =========================================================
option = st.selectbox(
    "📷 Select Input Type",
    [
        "Image",
        "Video",
        "Camera"
    ]
)

language = st.selectbox(
    "🌍 Select Language",
    [
        "English",
        "Hindi",
        "Bengali",
        "Punjabi",
        "Urdu",
        "Tamil"
    ]
)


city = st.selectbox(
    "📍 Select Your City",
    [
        "Sasaram",
        "Patna",
        "Delhi",
        "Mumbai",
        "Kolkata",
        "Lucknow",
        "Bengaluru",
        "Hyderabad",
        "Chennai",
        "Pune",
        "Jaipur",
        "Bhopal",
        "Ranchi",
        "Varanasi",
        "Prayagraj"
    ]
)


# =========================================================
# GET WEATHER
# =========================================================
weather = None

if city.strip():

    with st.spinner(
        "🌦 Getting weather information..."
    ):

        weather = fetch_weather(
            city
        )


# =========================================================
# IMAGE DETECTION
# =========================================================
if option == "Image":

    st.subheader(
        "🖼 Upload Plant Leaf Image"
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        try:

            # Read file bytes
            file_bytes = uploaded_file.getvalue()

            # Unique ID for history
            image_id = hashlib.sha256(
                file_bytes
            ).hexdigest()

            image = Image.open(
                uploaded_file
            ).convert(
                "RGB"
            )

            frame = np.array(
                image
            )

            st.image(
                frame,
                caption="Uploaded Image",
                width=500
            )

            with st.spinner(
                "🤖 AI is analyzing the image..."
            ):

                detection = coco_detect(
                    frame
                )

            # -----------------------------------------
            # NON-LEAF OBJECT DETECTED
            # -----------------------------------------
            if detection is not None:

                label, box, yolo_confidence = detection

                st.error(
                    f"❌ Possible non-leaf object detected: "
                    f"**{label.upper()}** "
                    f"({yolo_confidence * 100:.2f}%)"
                )

                x1, y1, x2, y2 = map(
                    int,
                    box
                )

                annotated_frame = frame.copy()

                # RGB red
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

                cv2.putText(
                    annotated_frame,
                    f"Detected: {label}",

                    (
                        x1,
                        max(
                            y1 - 10,
                            20
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.7,

                    (255, 0, 0),

                    2
                )

                st.image(
                    annotated_frame,
                    caption=(
                        f"Detected Object: {label}"
                    ),
                    width=700
                )

                st.warning(
                    "Please upload a clear image "
                    "containing mainly a plant leaf."
                )

            # -----------------------------------------
            # CNN DISEASE PREDICTION
            # -----------------------------------------
            else:

                with st.spinner(
                    "🌿 Detecting plant disease..."
                ):

                    plant, disease, conf = cnn_predict(
                        frame
                    )

                add_to_history(
                    image_id=image_id,

                    plant=plant,

                    disease=disease,

                    confidence=conf
                )

                show_prediction_result(
                    frame=frame,

                    plant=plant,

                    disease=disease,

                    confidence=conf,

                    weather=weather,

                    selected_language=language
                )

        except Exception as error:

            st.error(
                "❌ Image processing failed."
            )

            st.exception(
                error
            )


# =========================================================
# VIDEO DETECTION
# =========================================================
elif option == "Video":

    st.subheader(
        "🎥 Upload Plant Video"
    )

    video_file = st.file_uploader(
        "Choose a video",
        type=[
            "mp4",
            "avi",
            "mov"
        ]
    )

    if video_file is not None:

        temp_file = None

        try:

            # Save uploaded video temporarily
            suffix = os.path.splitext(
                video_file.name
            )[1]

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            )

            temp_file.write(
                video_file.read()
            )

            temp_file.close()

            cap = cv2.VideoCapture(
                temp_file.name
            )

            if not cap.isOpened():

                st.error(
                    "❌ Could not open video."
                )

            else:

                stframe = st.empty()

                status_placeholder = st.empty()

                frame_number = 0

                while cap.isOpened():

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_number += 1

                    detection = coco_detect(
                        frame
                    )

                    # ---------------------------------
                    # YOLO OBJECT DETECTION
                    # ---------------------------------
                    if detection is not None:

                        label, box, yolo_conf = detection

                        x1, y1, x2, y2 = map(
                            int,
                            box
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2
                        )

                        cv2.putText(
                            frame,

                            (
                                f"Possible Non-Leaf: "
                                f"{label}"
                            ),

                            (
                                x1,
                                max(
                                    y1 - 10,
                                    20
                                )
                            ),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.7,

                            (0, 0, 255),

                            2
                        )

                        status_placeholder.error(
                            f"Detected: {label}"
                        )

                    # ---------------------------------
                    # CNN PREDICTION
                    # ---------------------------------
                    else:

                        # OpenCV gives BGR,
                        # CNN requires RGB
                        rgb_frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB
                        )

                        plant, disease, conf = cnn_predict(
                            rgb_frame
                        )

                        prediction_text = (
                            f"{pretty_name(plant)} | "
                            f"{pretty_name(disease)} | "
                            f"{conf * 100:.1f}%"
                        )

                        cv2.putText(
                            frame,

                            prediction_text,

                            (20, 40),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.7,

                            (0, 255, 0),

                            2
                        )

                        status_placeholder.success(
                            prediction_text
                        )

                    stframe.image(
                        frame,
                        channels="BGR",
                        width=700
                    )

                cap.release()

                st.success(
                    "✅ Video analysis completed."
                )

        except Exception as error:

            st.error(
                "❌ Video processing failed."
            )

            st.exception(
                error
            )

        finally:

            if (
                temp_file is not None
                and
                os.path.exists(
                    temp_file.name
                )
            ):

                try:

                    os.remove(
                        temp_file.name
                    )

                except PermissionError:
                    pass


# =========================================================
# CAMERA DETECTION
# =========================================================
elif option == "Camera":

    st.subheader(
        "📷 Live Camera Detection"
    )

    st.info(
        "Start Camera checkbox ko enable karo. "
        "Stop karne ke liye checkbox ko disable karo."
    )

    run_camera = st.checkbox(
        "▶ Start Camera"
    )

    camera_placeholder = st.empty()

    prediction_placeholder = st.empty()

    if run_camera:

        # Windows webcam compatibility
        if os.name == "nt":

            cap = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW
            )

        else:

            cap = cv2.VideoCapture(
                0
            )

        if not cap.isOpened():

            st.error(
                "❌ Camera open nahi ho raha."
            )

            st.warning(
                "Check camera permission and "
                "close other apps using the camera."
            )

        else:

            while run_camera:

                ret, frame = cap.read()

                if not ret:

                    st.error(
                        "❌ Camera frame read nahi ho raha."
                    )

                    break

                detection = coco_detect(
                    frame
                )

                # ---------------------------------
                # NON-LEAF DETECTION
                # ---------------------------------
                if detection is not None:

                    label, box, yolo_conf = detection

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,

                        f"Possible Non-Leaf: {label}",

                        (
                            x1,
                            max(
                                y1 - 10,
                                20
                            )
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        (0, 0, 255),

                        2
                    )

                    prediction_placeholder.error(
                        f"Possible non-leaf object: "
                        f"{label}"
                    )

                # ---------------------------------
                # CNN DISEASE PREDICTION
                # ---------------------------------
                else:

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    plant, disease, conf = cnn_predict(
                        rgb_frame
                    )

                    prediction_text = (
                        f"{pretty_name(plant)} | "
                        f"{pretty_name(disease)} | "
                        f"{conf * 100:.1f}%"
                    )

                    cv2.putText(
                        frame,

                        prediction_text,

                        (20, 40),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        (0, 255, 0),

                        2
                    )

                    prediction_placeholder.success(
                        prediction_text
                    )

                camera_placeholder.image(
                    frame,
                    channels="BGR",
                    width=700
                )

                # Reduce CPU usage
                time.sleep(
                    0.03
                )

            cap.release()


# =========================================================
# SIDEBAR HISTORY
# =========================================================
with history_container:

    st.subheader(
        "🕒 Prediction History"
    )

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history[::-1]
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button(
            "🗑 Clear History"
        ):

            st.session_state.history = []

            st.session_state.processed_images = []

            st.rerun()

    else:

        st.info(
            "No predictions yet."
        )