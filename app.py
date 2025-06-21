import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image # For image manipulation if needed
import os

# --- Model Architecture (Must match the training script) ---
latent_dim = 100
num_classes = 10
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

# --- Load the trained model ---
@st.cache_resource # Cache the model loading for performance
def load_generator_model():
    generator = Generator()
    model_path = "models/generator_mnist_cgan.pth"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        st.stop()
    try:
        generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        generator.eval() # Set to evaluation mode
        st.success("Model loaded successfully!")
        return generator
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

generator = load_generator_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)


# --- Image Generation Function ---
def generate_images(model, digit, num_images=5):
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim, device=device)
        labels = torch.full((num_images,), digit, dtype=torch.long, device=device)
        generated_images = model(z, labels).cpu().numpy()

    # Rescale images from [-1, 1] to [0, 1] and then to [0, 255] for display
    generated_images = (generated_images + 1) / 2
    generated_images = (generated_images * 255).astype(np.uint8)

    # Convert from (N, C, H, W) to (N, H, W, C) for Streamlit's st.image
    if channels == 1:
        generated_images = np.transpose(generated_images, (0, 2, 3, 1))
        # Remove the channel dimension for grayscale images for proper display by st.image
        generated_images = generated_images.squeeze(-1)
    else:
        generated_images = np.transpose(generated_images, (0, 2, 3, 1))
    
    return generated_images

# --- Streamlit App Interface ---
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Generator")
st.markdown("Generate images of handwritten digits (0-9) using a trained Conditional GAN model.")

# User selects the digit
selected_digit = st.slider("Select a digit to generate:", 0, 9, 0)

# Button to generate images
if st.button("Generate 5 Images"):
    st.subheader(f"Generating 5 images for digit: {selected_digit}")
    
    generated_imgs = generate_images(generator, selected_digit, num_images=5)
    
    # Display images in a grid format
    cols = st.columns(5)
    for i, img_array in enumerate(generated_imgs):
        with cols[i]:
            st.image(img_array, caption=f"Digit {selected_digit}", use_column_width=True)

st.markdown("""
---
**How it works:**
This web app uses a Generative Adversarial Network (GAN) trained on the MNIST dataset. 
You select a digit, and the model generates 5 unique (but similar) images of that digit.
""")