import json

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ----------------------------------------------------------------------
# Page Setup & Styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="NutriVision — Food & Calorie AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design and cards
st.markdown(
    """
    <style>
    /* Global background and font improvements */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Custom Card Containers */
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    .calorie-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.3);
    }
    
    .calorie-box h3 {
        color: white !important;
        margin: 0;
        font-weight: 700;
    }
    
    .calorie-box p {
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 0;
    }

    /* Progress bar color customization */
    .stProgress > div > div > div > div {
        background-color: #11998e;
    }
    </style>
""",
    unsafe_allow_html=True,
)

MODEL_PATH = "food_classifier.pth"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CALORIE_DATA = {
    "pizza": {"serving": "1 slice (~107g)", "calories": 285},
    "sushi": {"serving": "6 pieces (~230g)", "calories": 255},
    "hamburger": {"serving": "1 regular burger (~110g)", "calories": 354},
    "ramen": {"serving": "1 bowl (~500g)", "calories": 436},
    "fried_rice": {"serving": "1 cup (~198g)", "calories": 333},
    "ice_cream": {"serving": "1/2 cup (~66g)", "calories": 137},
    "steak": {"serving": "6 oz grilled (~170g)", "calories": 330},
    "caesar_salad": {"serving": "1 bowl w/ dressing (~300g)", "calories": 470},
    "chicken_wings": {"serving": "6 wings (~150g)", "calories": 430},
    "french_fries": {"serving": "medium serving (~117g)", "calories": 365},
    "donuts": {"serving": "1 glazed (~60g)", "calories": 260},
    "pad_thai": {"serving": "1 serving (~400g)", "calories": 400},
}


def pretty_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


# ----------------------------------------------------------------------
# Model Loading & Inference
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, class_names


transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def predict(model, class_names, image: Image.Image):
    img = image.convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    k = min(3, len(class_names))
    top_probs, top_idx = torch.topk(probs, k=k)
    return [(class_names[int(i)], float(p)) for p, i in zip(top_probs, top_idx)]


# ----------------------------------------------------------------------
# Sidebar Controls & Information
# ----------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1498837167922-ddd27525d352?auto=format&fit=crop&w=600&q=80",
        use_container_width=True,
    )
    st.title("🥗 NutriVision")
    st.caption("BIT4443 Deep Learning Project")

    st.markdown("---")
    st.markdown("### 📋 Supported Dishes")
    for dish in sorted([pretty_name(d) for d in CALORIE_DATA.keys()]):
        st.markdown(f"• {dish}")

    st.markdown("---")
    st.markdown(
        "<small>Built with Streamlit & PyTorch • EfficientNet-B0</small>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Main Application Layout
# ----------------------------------------------------------------------
st.title("🍽️ Instant Food Classifier & Calorie Estimate")
st.write(
    "Snap or upload a photo of your meal to identify the dish and view nutritional estimates."
)

tab_try, tab_about, tab_model = st.tabs(
    ["🔍 Analyzer", "ℹ️ Project Overview", "🧠 Model Architecture"]
)

# ---- Tab 1: Analyzer ----
with tab_try:
    model_ready = False
    try:
        model, class_names = load_model()
        model_ready = True
    except FileNotFoundError:
        st.error(
            "⚠️ **Model files missing.** Please ensure `food_classifier.pth` and `class_names.json` are placed in the root directory."
        )

    # 2-Column Responsive Grid
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 1. Upload Photo")
        uploaded_file = st.file_uploader(
            "Select an image file",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(
                image,
                caption="Uploaded Meal",
                use_container_width=True,
            )

    with col_right:
        st.markdown("#### 2. Analysis & Results")

        if uploaded_file is None:
            st.info("👈 Upload a meal photo to see results.")
        elif model_ready:
            # Added Predict Button Here
            if st.button("Predict Dish & Calories", type="primary", use_container_width=True):
                with st.spinner("Analyzing image features..."):
                    results = predict(model, class_names, image)

                top_class, top_conf = results[0]

                # Top Prediction Highlight Card
                st.markdown(
                    f"""
                <div class="card">
                    <span style="color:#6c757d; font-size: 0.9em;">DETECTED DISH</span>
                    <h2 style="margin:0; color:#1a1a1a;">{pretty_name(top_class)}</h2>
                    <h3 style="margin:5px 0 0 0; color:#11998e;">{top_conf * 100:.1f}% <span style="font-size: 0.6em; color:#6c757d;">confidence</span></h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Calorie Info Box
                if top_class in CALORIE_DATA:
                    info = CALORIE_DATA[top_class]
                    st.markdown(
                        f"""
                    <div class="calorie-box">
                        <h3>🔥 ~{info['calories']} kcal</h3>
                        <p style="margin-top: 5px;">Serving size: <b>{info['serving']}</b></p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.caption(
                        "Note: Estimates are based on standard portions and may vary by recipe."
                    )

                st.write("")
                st.markdown("##### Probabilities")
                for cls, conf in results:
                    st.write(f"**{pretty_name(cls)}** ({conf * 100:.1f}%)")
                    st.progress(conf)

# ---- Tab 2: About ----
with tab_about:
    col_a, col_b = st.columns(2, gap="medium")
    with col_a:
        st.markdown("### Problem Statement")
        st.write(
            "Dietary tracking often fails due to the friction of manually searching calorie databases. "
            "NutriVision removes manual entry by converting dish visual data directly into calorie estimates."
        )

        st.markdown("### Target Audience")
        st.write(
            "Fitness enthusiasts, individuals tracking macronutrients, and digital health applications requiring quick meal logging."
        )

    with col_b:
        st.markdown("### Scope & Capabilities")
        st.write(
            "Recognizes 12 core dietary categories with pre-computed nutritional serving data tuned for common consumer portion sizes."
        )

# ---- Tab 3: Model Info ----
with tab_model:
    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("### Architecture")
        st.write(
            "• **Backbone:** Fine-tuned EfficientNet-B0 pretrained on ImageNet\n"
            "• **Optimization:** Two-stage transfer learning (Feature extraction classifier head + block fine-tuning)\n"
            "• **Input Resolution:** 224x224 RGB"
        )

    with m_col2:
        st.markdown("### Dataset Details")
        st.write(
            "• **Source:** Food-101 Subset\n"
            "• **Split:** 750 training images / 250 evaluation images per dish\n"
            "• **Total Classes:** 12 food types"
        )