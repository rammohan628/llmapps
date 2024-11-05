import streamlit as st
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

#Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#Streamlit app title
st.title("CLIP Image-Text Similarity Predictor")

# Upload image section
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Input text description
text_inputs = st.text_input("Enter text descriptions (comma-separated):", "a photo of a cat, a photo of a dog")

# When the user clicks the predict button
if st.button("Predict"):
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Process the inputs
        text_descriptions = [text.strip() for text in text_inputs.split(',')]
        inputs = processor(text=text_descriptions, images=image, return_tensors="pt", padding=True)
        
        # Get the model outputs
        with torch.no_grad():  # Disable gradient calculations for inference
            outputs = model(**inputs)
        
        # Get similarity scores
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1)  # Probabilities

        # Display the results
        st.subheader("Predictions:")
        for description, prob in zip(text_descriptions, probs[0]):
            st.write(f"**{description}**: {prob:.4f}")
    else:
        st.warning("Please upload an image to get predictions.")

