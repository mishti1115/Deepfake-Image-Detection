import streamlit as st
import torch
import csv
import os
import requests

from PIL import Image
from io import BytesIO
from transformers import AutoImageProcessor, SiglipForImageClassification


MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"
CSV_FILE = "deepfake_sonuclar.csv"


legal_information = {
    "India": (
        "Deepfake content may be subject to applicable Indian laws. "
        "The legal position depends on the circumstances."
    ),
    "USA": (
        "Deepfake content may be subject to applicable federal and state laws."
    ),
    "Germany": (
        "Deepfake content may be subject to applicable German and EU laws."
    ),
    "France": (
        "Deepfake content may be subject to applicable French and EU laws."
    ),
    "Türkiye": (
        "Deepfake content may be subject to applicable Turkish laws."
    )
}


@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    return processor, model


processor, model = load_model()


def save_results(source, fake_score, real_score, classification):
    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Image Source",
                "Fake Score",
                "Real Score",
                "Classification"
            ])

        writer.writerow([
            source,
            f"{fake_score:.4f}",
            f"{real_score:.4f}",
            classification
        ])


def get_image(uploaded_file, image_url):
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        return image, uploaded_file.name

    if image_url.strip():
        response = requests.get(
            image_url.strip(),
            timeout=20
        )
        response.raise_for_status()

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        return image, image_url.strip()

    return None, None


def predict_image(image):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

    result = []

    for index, score in enumerate(probabilities):
        label = model.config.id2label.get(
            index,
            f"Class {index}"
        )

        result.append({
            "label": str(label),
            "score": float(score.item()),
            "class_id": index
        })

    return result


def main():
    st.set_page_config(
        page_title="Deepfake Detection System",
        page_icon="🔍",
        layout="centered"
    )

    st.title("Deepfake Detection System")

    st.write(
        "Upload an image to check the model's prediction "
        "about whether it may be real or manipulated."
    )

    selected_country = st.sidebar.selectbox(
        "Select Country for Legal Information",
        list(legal_information.keys())
    )

    st.sidebar.info(
        legal_information[selected_country]
    )

    st.subheader("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    image_url = st.text_input(
        "Or enter an image URL"
    )

    if st.button("Analyze Image"):
        try:
            image, source = get_image(
                uploaded_file,
                image_url
            )

            if image is None:
                st.warning(
                    "Please upload an image or enter an image URL."
                )
                return

            st.image(
                image,
                caption="Image to Analyze",
                use_container_width=True
            )

            with st.spinner("Analyzing image..."):
                result = predict_image(image)

            st.subheader("Detection Result")

            for item in result:
                st.write(
                    f"{item['label']}: "
                    f"{item['score'] * 100:.2f}% confidence"
                )

            st.write("DEBUG OUTPUT:", result)

            fake_score = None
            real_score = None

            for item in result:
                label = item["label"].strip().lower()

                if "fake" in label:
                    fake_score = item["score"]

                elif "real" in label:
                    real_score = item["score"]

            if fake_score is None or real_score is None:
                st.warning(
                    "The model's labels could not be mapped "
                    "to REAL and FAKE safely. Please check "
                    "the DEBUG OUTPUT."
                )
                return

            st.divider()

            if fake_score > real_score:
                classification = "FAKE prediction"

                st.error(
                    "⚠️ The model predicts that this image may be FAKE."
                )

            elif real_score > fake_score:
                classification = "REAL prediction"

                st.success(
                    "✅ The model predicts that this image may be REAL."
                )

            else:
                classification = "Uncertain prediction"

                st.warning(
                    "The model gave equal scores for REAL and FAKE."
                )

            st.caption(
                "These are model confidence scores, not proof "
                "that an image is real or fake."
            )

            save_results(
                source,
                fake_score,
                real_score,
                classification
            )

            st.success("Result saved successfully.")

        except Exception as e:
            st.error(
                f"Could not analyze the image: {e}"
            )


if __name__ == "__main__":
    main()
