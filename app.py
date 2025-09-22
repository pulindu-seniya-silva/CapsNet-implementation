# app.py — Streamlit UI for CapsNet using weights-only loading
# Clear prediction banner + Top-3 table + debug info

import os
import numpy as np
from PIL import Image
import streamlit as st

# ---------- Pick a Keras flavor dynamically ----------
USING_TF_KERAS = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    USING_TF_KERAS = True
except Exception:
    import tensorflow as tf
    from keras import layers, Model, Input

# ---------- CapsNet building blocks (match training code) ----------
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1.0 + s_squared_norm)
    return scale * vectors / tf.sqrt(s_squared_norm + 1e-8)

def ConvLayer(inputs):
    return layers.Conv2D(256, kernel_size=9, strides=1, activation='relu')(inputs)

def PrimaryCaps(inputs, dim_caps=8, n_channels=32, kernel_size=9, strides=2):
    conv = layers.Conv2D(filters=dim_caps * n_channels,
                         kernel_size=kernel_size,
                         strides=strides,
                         activation='relu')(inputs)
    capsules = layers.Reshape((-1, dim_caps))(conv)
    capsules = layers.Lambda(squash)(capsules)
    return capsules

class DigitCaps(layers.Layer):
    def __init__(self, num_caps=10, dim_caps=16, routing_iters=3, **kwargs):
        super().__init__(**kwargs)
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.input_num_caps = input_shape[1]
        self.input_dim_caps = input_shape[2]
        self.W = self.add_weight(
            shape=[1, self.input_num_caps, self.num_caps, self.dim_caps, self.input_dim_caps],
            initializer='glorot_uniform',
            trainable=True,
            name="W_caps"
        )
        super().build(input_shape)

    def call(self, u):
        batch_size = tf.shape(u)[0]
        u = tf.expand_dims(tf.expand_dims(u, 2), -1)   # [B, in_caps, 1, dim_in, 1]
        W_tiled = tf.tile(self.W, [batch_size, 1, 1, 1, 1])
        u_hat = tf.matmul(W_tiled, u)                  # [B, in_caps, num_caps, dim_caps, 1]
        u_hat = tf.squeeze(u_hat, axis=-1)             # [B, in_caps, num_caps, dim_caps]

        b = tf.zeros([batch_size, self.input_num_caps, self.num_caps])
        for i in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)
            s = tf.reduce_sum(tf.expand_dims(c, -1) * u_hat, axis=1)
            v = squash(s)
            if i < self.routing_iters - 1:
                b += tf.reduce_sum(u_hat * tf.expand_dims(v, 1), axis=-1)
        return v

def capsule_length(z):
    return tf.sqrt(tf.reduce_sum(tf.square(z), axis=-1))

def build_capsnet_model(input_shape=(28, 28, 1), num_classes=10):
    inputs = Input(shape=input_shape, name="input_image")
    x = ConvLayer(inputs)
    x = PrimaryCaps(x)
    digit_caps = DigitCaps()(x)
    output = layers.Lambda(capsule_length, name="capsule_length")(digit_caps)
    return Model(inputs=inputs, outputs=output, name="CapsNet")

# ---------- App config ----------
WEIGHTS_PATH = "capsnet.weights.h5"
INPUT_SIZE = (28, 28)                  # MNIST
CHANNELS = 1                           # grayscale
CLASS_NAMES = [str(i) for i in range(10)]
CONFIDENCE_WARN_THRESHOLD = 0.40       # tweak as you like

st.set_page_config(page_title="Capsule Network — Real-time Inference", page_icon="🧠", layout="centered")
st.title("🧠 Capsule Network — Real-time Inference")
st.caption(("Using tensorflow.keras" if USING_TF_KERAS else "Using standalone keras")
           + " • Upload a digit image (MNIST-like).")

# ---------- Build model & load weights ----------
@st.cache_resource(show_spinner=True)
def load_model_with_weights():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing '{WEIGHTS_PATH}' next to app.py")
    model = build_capsnet_model(input_shape=(28, 28, 1), num_classes=10)
    model.load_weights(WEIGHTS_PATH)
    return model

try:
    model = load_model_with_weights()
except Exception as e:
    st.error(f"Failed to initialize model from weights.\n\n{e}")
    st.stop()

# ---------- Preprocess ----------
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("L" if CHANNELS == 1 else "RGB")
    img = img.resize(INPUT_SIZE, Image.LANCZOS)
    x = np.asarray(img, dtype=np.float32) / 255.0
    if CHANNELS == 1:
        x = np.expand_dims(x, axis=-1)   # (H, W, 1)
    x = np.expand_dims(x, axis=0)        # (1, H, W, C)
    return x

# ---------- Sidebar (debug) ----------
with st.sidebar:
    st.header("Settings")
    invert = st.checkbox("Invert colors (white digit on black)", value=False)
    st.caption(f"Weights: `{WEIGHTS_PATH}`")

# ---------- Input ----------
uploaded = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"])
st.markdown("Or take a quick photo:")
cam = st.camera_input("Camera (hold up a digit)")

img = None
if uploaded:
    img = Image.open(uploaded).convert("RGB")
elif cam:
    img = Image.open(cam).convert("RGB")

if img is None:
    st.info("Upload an image or use the camera to start.")
    st.stop()

st.subheader("Input Preview")
st.image(img, width=256)

if invert:
    img = Image.fromarray(255 - np.array(img))

# ---------- Predict ----------
x = preprocess(img)
preds = model(x)              # shape (1, 10) — capsule lengths
scores = np.array(preds)[0]   # shape (10,)

# Normalize just for display (capsule lengths aren't probabilities)
probs = scores / (scores.sum() + 1e-8)

top_idx = int(np.argmax(scores))
top_label = CLASS_NAMES[top_idx]
top_conf = float(probs[top_idx])

# ---------- Big, explicit result ----------
if top_conf < CONFIDENCE_WARN_THRESHOLD:
    st.warning(f"Low confidence. Highest score → **{top_label}** ({top_conf*100:.1f}%). "
               "Try toggling **Invert colors** or provide a clearer MNIST-like digit.")
else:
    st.success(f"**Prediction: {top_label}**  \nConfidence (normalized): **{top_conf*100:.1f}%**")

# ---------- Top-3 table ----------
top3_idx = np.argsort(scores)[::-1][:3]
top3 = [(CLASS_NAMES[i], float(probs[i]), float(scores[i])) for i in top3_idx]
st.subheader("Top-3 classes")
st.table(
    [{"class": c, "confidence(%)": f"{p*100:.2f}", "raw score": f"{s:.4f}"} for c, p, s in top3]
)

# ---------- Bar chart ----------
st.subheader("All class scores")
st.bar_chart({"class": CLASS_NAMES, "score": probs.tolist()}, x="class", y="score", height=240)

# ---------- Debug raw scores in sidebar ----------
with st.sidebar:
    st.markdown("---")
    st.subheader("Raw scores")
    st.json({CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))})
