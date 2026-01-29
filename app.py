import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

# =========================================================
# CONFIGURATION
# =========================================================
IMAGE_SIZE = 256
CHANNELS = 1
MODEL_PATH = "dncnn_medical_model.h5"

st.set_page_config(
    page_title="Medical Image Denoising",
    layout="wide"
)

# =========================================================
# DnCNN MODEL
# =========================================================
from tensorflow.keras.layers import Lambda

def build_dncnn(depth=17, filters=64):
    inp = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

    x = Conv2D(filters, 3, padding="same")(inp)
    x = ReLU()(x)

    for _ in range(depth - 2):
        x = Conv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    noise = Conv2D(1, 3, padding="same")(x)

    # Explicit connection for Keras 3 compatibility
    out = Lambda(lambda z: z)(noise)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


# =========================================================
# IMAGE PROCESSING
# =========================================================
def preprocess_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    return img.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)

def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma / 255.0, image.shape)
    return np.clip(image + noise, 0, 1)

def denoise_image(model, noisy_img):
    noisy = noisy_img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    predicted_noise = model.predict(noisy, verbose=0)
    denoised = noisy - predicted_noise
    return np.clip(denoised, 0, 1).reshape(IMAGE_SIZE, IMAGE_SIZE)

def compute_metrics(clean, denoised):
    psnr = peak_signal_noise_ratio(clean, denoised, data_range=1.0)
    ssim = structural_similarity(clean, denoised, data_range=1.0)
    mse = mean_squared_error(clean, denoised)
    return psnr, ssim, mse

# =========================================================
# LOAD OR CREATE MODEL
# =========================================================
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    st.sidebar.success("Pre-trained Model Loaded")
else:
    model = build_dncnn()
    st.sidebar.warning("Using Untrained Model (Demo Mode)")

# =========================================================
# UI
# =========================================================
st.title("Medical Image Denoising Using Deep Learning")
st.write("DnCNN-based Medical Image Enhancement Web Application")

st.sidebar.header("Settings")
noise_level = st.sidebar.slider("Gaussian Noise Level", 5, 50, 25)
show_histogram = st.sidebar.checkbox("Show Histogram", True)

uploaded_file = st.file_uploader(
    "Upload Medical Image (MRI / CT / X-ray)",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PROCESS IMAGE
# =========================================================
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    clean_img = preprocess_image(image).squeeze()
    noisy_img = add_gaussian_noise(clean_img.reshape(IMAGE_SIZE, IMAGE_SIZE, 1), noise_level)
    denoised_img = denoise_image(model, noisy_img.squeeze())

    psnr, ssim, mse = compute_metrics(clean_img, denoised_img)

    st.subheader("Denoising Results")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(clean_img, caption="Original Image", clamp=True, use_container_width=True)
    with c2:
        st.image(noisy_img.squeeze(), caption=f"Noisy Image (σ={noise_level})", clamp=True, use_container_width=True)
    with c3:
        st.image(denoised_img, caption="Denoised Output", clamp=True, use_container_width=True)

    st.subheader("Image Quality Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("PSNR (dB)", f"{psnr:.2f}")
    m2.metric("SSIM", f"{ssim:.4f}")
    m3.metric("MSE", f"{mse:.6f}")

    if show_histogram:
        st.subheader("Pixel Intensity Distribution")
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))

        ax[0].hist(clean_img.ravel(), bins=50)
        ax[0].set_title("Original")

        ax[1].hist(noisy_img.ravel(), bins=50)
        ax[1].set_title("Noisy")

        ax[2].hist(denoised_img.ravel(), bins=50)
        ax[2].set_title("Denoised")

        st.pyplot(fig)
