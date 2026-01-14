import streamlit as st 
from matplotlib import pylab as plt 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 
import numpy as np
import seaborn as sns



@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_papers():
    return pd.read_csv('/Users/chandlershortlidge/Desktop/Ironhack/end-to-end-project/notebooks/papers.csv')

@st.cache_data
def load_embeddings():
    return np.load("/Users/chandlershortlidge/Desktop/Ironhack/end-to-end-project/notebooks/embeddings.npy", allow_pickle=True)

@st.cache_data
def load_aqi_data():
    return pd.read_csv("/Users/chandlershortlidge/Desktop/Ironhack/end-to-end-project/notebooks/processed_aqi_data.csv")


def main():
    # homepage
    st.title("Dangerous Air Quality Info") 
    st.image("/Users/chandlershortlidge/Desktop/Ironhack/end-to-end-project/images/Screenshot 2026-01-13 at 14.45.37.png")



    st.header("Effects of Air Polution")
    st.image("/Users/chandlershortlidge/Desktop/Ironhack/end-to-end-project/images/air polution.jpeg")

    st.header("Effects of PM2.5")
    st.image("/Users/chandlershortlidge/Desktop/Ironhack/end-to-end-project/images/Screenshot 2026-01-13 at 15.47.18.png")


# SEARCH FUNCTION    
    papers = load_papers()


   # Load model and encode descriptions (cached)
    model = load_model()
    embeddings = load_embeddings()
    #UI
    st.header("Learn more about AQI and your health")
    user_input = st.text_area("What is your question?")

    if st.button("Search"):
        # encodes single user input to 1 vector
        user_embedding = model.encode([user_input])
        # Compute cosine similarities
        similarities = cosine_similarity(user_embedding, embeddings)
                    # compares that one user vector against all 1000 and returns a similarity score for each.
        # Get indices of top 3 highest scores
        top_indices = similarities[0].argsort()[-3:][::-1]
        # argsort() returns: [0, 1, 2, 3, 4] (indices in order of score, low to high)
        # [-3:] = "last 3 items" --> [2, 3, 4] (the 3 highest scores)
        # [::-1] = "reverse it" --> [4, 3, 2] (now highest first)

        # Use those indices to get the films
        top_papers = papers.iloc[top_indices]
        # iloc lets you select rows by their index position (0, 1, 2, etc.).
        # So if top_3_indices is [633, 292, 845] then films.iloc[top_3_indices] 
        # Grabs rows 633, 292, and 845 from the films DataFrame — which are your top 3 matching movies.

        st.write("Top 3 Matches:")
        st.dataframe(top_papers[['title', 'abstract']])































if __name__ == '__main__':
    main()


