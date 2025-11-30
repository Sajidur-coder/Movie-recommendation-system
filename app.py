import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------ LOAD DATA ------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    tags = pd.read_csv("tags.csv")

    # Merge all tags per movie
    tags_group = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x))

    # Add tags to movie table
    movies = movies.merge(tags_group, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")

    # Combine genres + tags for similarity
    movies["combined_features"] = movies["genres"] + " " + movies["tag"]

    return movies


movies = load_data()


# ------------ NORMALIZE ------------
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------ FIND MOVIE BY PARTIAL MATCH ------------
def find_movie(title):
    title_norm = normalize(title)
    movies["norm_title"] = movies["title"].apply(normalize)
    matches = movies[movies["norm_title"].str.contains(title_norm)]

    if matches.empty:
        return None

    return matches.iloc[0]["title"]


# ------------ TF-IDF MATRIX + SIMILARITY ------------
@st.cache_resource
def build_similarity(movies):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

    similarity = cosine_similarity(tfidf_matrix)
    return similarity


content_similarity = build_similarity(movies)


# ------------ CONTENT-BASED RECOMMENDATION ------------
def recommend_content(title, n=10):
    real_title = find_movie(title)

    if real_title is None:
        return ["Movie not found!"]

    idx = movies[movies["title"] == real_title].index[0]

    scores = list(enumerate(content_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top = scores[1:n+1]
    recs = movies.iloc[[i[0] for i in top]]["title"].values
    return recs


# ------------ STREAMLIT UI ------------
st.title("🎬 Movie Recommendation System (Content-Based)")
st.write("This system recommends movies based on genres and user-added tags.")

movie_name = st.text_input("Enter movie name:")

if st.button("Recommend"):
    st.write("### ⭐ Content-Based Recommendations:")
    for m in recommend_content(movie_name):
        st.write("→", m)
