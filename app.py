import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

MODEL_PATH = "animal_model.h5"   
DATASET_PATH = "dataset"         
IMG_SIZE = (224, 224)


st.set_page_config(page_title="Animal Classifier", layout="centered")

st.title("Animal Classifier AI")
st.write("Upload ảnh động vật để AI dự đoán")


@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

class_names = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])
import json

with open("animal_info.json", "r", encoding="utf-8") as f:
    animal_info = json.load(f)


uploaded_file = st.file_uploader(
    "Chọn ảnh", type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, width=300)

    # preprocess
    IMG_SIZE = (224, 224)
    img = img.resize(IMG_SIZE)

    img_array = np.array(img)
    img_array = img_array
    img_array = np.expand_dims(img_array, axis=0)


    predictions = model.predict(img_array)[0]

    max_prob = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    CONFIDENCE_THRESHOLD = 0.6

    st.subheader("Kết quả:")

    if max_prob < CONFIDENCE_THRESHOLD:
        st.error("❌ Không thể dự đoán (ảnh không thuộc dataset)")
    else:
        st.success(f"✅ {predicted_class}")
        st.write(f"Độ tin cậy: {max_prob:.2f}")

    if predicted_class in animal_info:
        info = animal_info[predicted_class]

        st.markdown("### 📚 Thông tin về loài:")

        st.write(f"**Tên:** {info['name']}")
        st.write(f"**Tên khoa học:** {info['scientific']}")
        st.write(f"**Môi trường sống:** {info['habitat']}")
        st.write(f"**Chế độ ăn:** {info['diet']}")
        st.write(f"**Fun fact:** {info['fact']}")
    else:
        st.info("Chưa có thông tin cho loài này.")

