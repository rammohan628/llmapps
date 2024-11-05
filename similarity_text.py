import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load pre-trained model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

st.title("Sentence Similarity App")

# User input section
st.write("Enter a source sentence and sentences to compare it to:")

# Input for the source sentence
source_sentence = st.text_input("Source Sentence")

# Input for comparison sentences
compare_sentences = st.text_area("Sentences to Compare (one per line)").splitlines()

# Button to compute similarity
if st.button("Compute") and source_sentence and compare_sentences:
    # Encode sentences
    source_embedding = model.encode([source_sentence], convert_to_tensor=True)
    compare_embeddings = model.encode(compare_sentences, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(source_embedding, compare_embeddings)[0]

    # Display results
    st.write("Similarity Scores:")
    for sentence, similarity in zip(compare_sentences, similarities):
        st.write(f"{sentence}: {similarity.item():.3f}")

