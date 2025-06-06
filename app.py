import streamlit as st
from PIL import Image
from pathlib import Path
import torch
import os
import sys
import requests
import traceback

# Add safetensors library for loading the new model format
from safetensors.torch import load_file

# Set page config as the FIRST Streamlit command
st.set_page_config(layout="wide", page_title="Stable Diffusion Hub")

# --- Configuration & Path Setup ---
BASE_DIR = Path(__file__).resolve().parent
PYTORCH_SD_DIR = BASE_DIR / "pytorch-stable-diffusion-main"
DATA_DIR = PYTORCH_SD_DIR / "data"
SD_CODE_DIR = PYTORCH_SD_DIR / "sd"

# Add the 'sd' directory to Python's path
if str(SD_CODE_DIR) not in sys.path:
    sys.path.append(str(SD_CODE_DIR))

# Attempt to import Stable Diffusion components
try:
    # We will need these individual components for our new loading function
    from clip import CLIP
    from encoder import VAE_Encoder
    from decoder import VAE_Decoder
    from diffusion import Diffusion
    # We still import the old ones for pipeline and tokenizer
    import model_loader
    import pipeline
    from transformers import CLIPTokenizer
    sd_components_available = True
except ImportError as e:
    st.error(f"Failed to import Stable Diffusion components: {e}")
    sd_components_available = False

# --- Model Configuration ---
DEVICE = "cpu"

# --- Function to Download Stable Diffusion Model ---
MODEL_URL = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
STABLE_DIFFUSION_MODEL_FILENAME = "v1-5-pruned-emaonly.safetensors"
MODEL_PATH_FOR_DOWNLOAD = DATA_DIR / STABLE_DIFFUSION_MODEL_FILENAME

@st.cache_resource
def download_sd_model_if_needed():
    """Downloads the Stable Diffusion model if it doesn't already exist."""
    if not MODEL_PATH_FOR_DOWNLOAD.exists():
        st.info(f"Downloading Stable Diffusion model ({STABLE_DIFFUSION_MODEL_FILENAME})...")
        MODEL_PATH_FOR_DOWNLOAD.parent.mkdir(parents=True, exist_ok=True)
        hf_token = st.secrets.get("HUGGINGFACE_TOKEN")
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        try:
            with requests.get(MODEL_URL, stream=True, headers=headers) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = st.progress(0, text=f"Downloading {STABLE_DIFFUSION_MODEL_FILENAME}...")
                bytes_downloaded = 0
                with open(MODEL_PATH_FOR_DOWNLOAD, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        if chunk:
                            f.write(chunk)
                            bytes_downloaded += len(chunk)
                            if total_size > 0:
                                progress_bar.progress(min(bytes_downloaded / total_size, 1.0))
                progress_bar.progress(1.0)
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            if MODEL_PATH_FOR_DOWNLOAD.exists():
                os.remove(MODEL_PATH_FOR_DOWNLOAD)
            return False
    return True

# --- NEW, ROBUST STABLE DIFFUSION MODEL LOADING FUNCTION ---
@st.cache_resource
def load_sd_models_cached():
    """
    This function now correctly handles .safetensors files by manually
    loading the state dictionary and assigning weights to each component.
    """
    model_available = download_sd_model_if_needed()
    if not model_available:
        return None, None
    if not sd_components_available:
        st.error("Stable Diffusion Python components are not loaded.")
        return None, None

    try:
        with st.spinner("Loading Stable Diffusion models into memory..."):
            # Load the state dictionary from the safetensors file
            model_file_path = MODEL_PATH_FOR_DOWNLOAD
            state_dict = load_file(model_file_path, device=DEVICE)

            # --- Instantiate each model component ---
            encoder = VAE_Encoder().to(DEVICE)
            decoder = VAE_Decoder().to(DEVICE)
            diffusion = Diffusion().to(DEVICE)
            clip = CLIP().to(DEVICE)

            # --- Load weights into each component ---
            # The keys in the state_dict have prefixes that need to be removed.
            # e.g., 'model.diffusion_model.conv_in.weight' -> 'conv_in.weight'

            # VAE Encoder
            encoder_weights = {k.replace("first_stage_model.encoder.", ""): v for k, v in state_dict.items() if k.startswith("first_stage_model.encoder.")}
            encoder.load_state_dict(encoder_weights)

            # VAE Decoder
            decoder_weights = {k.replace("first_stage_model.decoder.", ""): v for k, v in state_dict.items() if k.startswith("first_stage_model.decoder.")}
            decoder.load_state_dict(decoder_weights)

            # Diffusion (UNet)
            diffusion_weights = {k.replace("model.diffusion_model.", ""): v for k, v in state_dict.items() if k.startswith("model.diffusion_model.")}
            diffusion.load_state_dict(diffusion_weights)

            # CLIP
            clip_weights = {k.replace("cond_stage_model.transformer.", ""): v for k, v in state_dict.items() if k.startswith("cond_stage_model.transformer.")}
            clip.load_state_dict(clip_weights)

            # --- Assemble the final models dictionary ---
            models = {
                "clip": clip,
                "encoder": encoder,
                "decoder": decoder,
                "diffusion": diffusion,
            }

            # --- Load tokenizer separately ---
            tokenizer_path = DATA_DIR / "vocab.json"
            merges_path = DATA_DIR / "merges.txt"
            tokenizer = CLIPTokenizer(str(tokenizer_path), merges_file=str(merges_path))

        st.success("✅ Stable Diffusion models loaded successfully.")
        return tokenizer, models

    except Exception as e:
        st.error(f"❌ Error loading Stable Diffusion models: {e}")
        st.error(traceback.format_exc())
        return None, None

# --- Streamlit App UI ---
st.title("✨ Stable Diffusion Image Generation")
st.markdown("Enter a prompt or upload an image to generate a new image.")

if not sd_components_available:
    st.error("Stable Diffusion functionality is currently unavailable due to import errors.")
else:
    tokenizer, sd_models = load_sd_models_cached()

    if tokenizer and sd_models:
        st.sidebar.header("⚙️ Settings")
        # ...(The rest of your UI code is unchanged and should work perfectly)...
        sd_input_type = st.sidebar.radio(
            "Select Input Type:",
            ("Text-to-Image", "Image-to-Image", "Text-and-Image")
        )
        st.sidebar.markdown("---")

        st.subheader("✏️ Prompts & Guidance")
        col1, col2 = st.columns(2)
        with col1:
            prompt = st.text_area("Prompt:", "A photorealistic cat astronaut exploring a neon-lit alien jungle, cinematic lighting, 8k", height=100)
        with col2:
            uncond_prompt = st.text_area("Negative Prompt (Optional):", "blurry, low quality, watermark, text, signature, ugly, deformed", height=100)
        cfg_scale = st.slider("CFG Scale (Guidance Strength):", 1.0, 20.0, 7.5, 0.5)

        input_image_pil = None
        strength = 0.8
        if sd_input_type in ["Image-to-Image", "Text-and-Image"]:
            st.subheader("🖼️ Input Image")
            uploaded_file = st.file_uploader("Upload an Image:", type=["png", "jpg", "jpeg"])
            if uploaded_file:
                input_image_pil = Image.open(uploaded_file).convert("RGB")
                st.image(input_image_pil, caption="Uploaded Image", width=300, use_column_width='auto')
            strength = st.slider("Strength:", 0.0, 1.0, 0.8, 0.05)

        st.sidebar.markdown("---")
        st.sidebar.subheader("Sampling Parameters")
        sampler = st.sidebar.selectbox("Sampler:", ["ddpm", "ddim"], index=0)
        num_inference_steps = st.sidebar.slider("Inference Steps:", 10, 150, 50, 1)
        seed = st.sidebar.number_input("Seed:", value=42, step=1)

        st.markdown("---")
        if st.button("🚀 Generate Image", type="primary"):
            with st.spinner("⏳ Generating image... This might take a moment!"):
                try:
                    output_image_array = pipeline.generate(
                        prompt=prompt,
                        uncond_prompt=uncond_prompt,
                        input_image=input_image_pil,
                        strength=strength,
                        do_cfg=True,
                        cfg_scale=cfg_scale,
                        sampler_name=sampler,
                        n_inference_steps=num_inference_steps,
                        seed=seed,
                        models=sd_models,
                        device=DEVICE,
                        idle_device="cpu",
                        tokenizer=tokenizer,
                    )
                    final_image = Image.fromarray(output_image_array)
                    st.image(final_image, caption=f"Generated Image (Seed: {seed})", use_column_width=True)
                except Exception as e:
                    st.error(f"❌ An error occurred during generation: {e}")
                    st.error(traceback.format_exc())
    else:
        st.warning("Models could not be loaded. Please check the logs for errors.")
