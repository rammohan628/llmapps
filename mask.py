import streamlit as st
from transformers import pipeline

# Initialize the unmasker pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')

# Title of the Streamlit app
st.title("BERT Masked Language Model")

# Input text area for the user to enter their sentence
input_text = st.text_area("Enter a sentence with a [MASK] token:", "Hello I'm a [MASK] model.")

# When the user clicks the button
if st.button("Predict"):
    # Get predictions from the model
    predictions = unmasker(input_text)
    
    # Display the predictions
    st.subheader("Predictions:")
    for prediction in predictions:
        st.write(f"**{prediction['token_str']}**: {prediction['score']:.4f}")

