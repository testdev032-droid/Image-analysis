import base64
from io import BytesIO

import streamlit as st
from PIL import Image
from groq import Groq

import config


# ---------------------------------
# Page setup
# ---------------------------------
st.set_page_config(
    page_title="AI Visionary",
    page_icon="🕵️",
    layout="centered",
)

st.title("🕵️ The AI Visionary")
st.write("Upload an image and let AI create a fun report about it!")

st.markdown(
    """
Choose an image, pick a report style, and click **Analyze Image**.
The AI will study the image and write a detailed report.
"""
)

# ---------------------------------
# Create Groq client
# ---------------------------------
client = Groq(api_key=config.GROQ_API_KEY)


# ---------------------------------
# Helper functions
# ---------------------------------
def encode_image_to_base64(uploaded_file) -> str:
    """
    Convert the uploaded image into base64 text.
    Groq Vision can read images passed as a data URL.
    """
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")


def build_analysis_prompt(report_style: str) -> str:
    """
    Build a prompt based on the selected report style.
    """
    styles = {
        "Normal": (
            "Look at this image carefully and write a clear, detailed report. "
            "Describe the scene, objects, and what seems to be happening."
        ),
        "Funny": (
            "Look at this image carefully and write a funny image report. "
            "Mention objects, details, and make the report playful and humorous, "
            "but still describe the image correctly."
        ),
        "Detective": (
            "Look at this image like a detective. "
            "Write an investigation-style report with clues, observations, and smart deductions."
        ),
        "Dramatic": (
            "Look at this image and describe it in a dramatic, cinematic way. "
            "Make the report vivid, exciting, and expressive."
        ),
        "Story Mode": (
            "Look at this image and write a short story-like scene description. "
            "Describe the setting, objects, and mood in a creative way."
        ),
    }
    return styles.get(report_style, styles["Normal"])


def analyze_image_with_groq(uploaded_file, report_style: str) -> str:
    """
    Send the image + prompt to Groq Vision and return the report text.
    """
    base64_image = encode_image_to_base64(uploaded_file)
    mime_type = uploaded_file.type  # example: image/png or image/jpeg
    prompt_text = build_analysis_prompt(report_style)

    completion = client.chat.completions.create(
        model=config.GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        temperature=0.8,
        max_completion_tokens=500,
    )

    return completion.choices[0].message.content


# ---------------------------------
# UI
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp"],
)

report_style = st.selectbox(
    "Choose report style",
    ["Normal", "Funny", "Detective", "Dramatic", "Story Mode"],
)

if uploaded_file is not None:
    image = Image.open(BytesIO(uploaded_file.getvalue()))
    st.image(image, caption="Uploaded Image", use_container_width=True)

if st.button("🔍 Analyze Image"):
    if not config.GROQ_API_KEY:
        st.error("Groq API key is missing. Please add it to your .env file.")
        st.stop()

    if uploaded_file is None:
        st.warning("Please upload an image first.")
        st.stop()

    with st.spinner("The AI is studying your image..."):
        try:
            report = analyze_image_with_groq(uploaded_file, report_style)
            st.success("Report ready!")
            st.subheader("📝 AI Report")
            st.write(report)
        except Exception as error:
            st.error(f"Something went wrong: {error}")