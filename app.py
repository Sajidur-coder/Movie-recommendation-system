import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")

tags_group = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x))
movies = movies.merge(tags_group, on="movieId", how="left")
movies["tag"] = movies["tag"].fillna("")
movies["combined_features"] = movies["genres"] + " " + movies["tag"]



with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)   
    text = re.sub(r"\s+", " ", text).strip()  
    return text


def find_movie(title):
    title_norm = normalize(title)

    movies["norm_title"] = movies["title"].apply(normalize)

    matches = movies[movies["norm_title"].str.contains(title_norm)]

    if matches.empty:
        return None

    return matches.iloc[0]["title"]


tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined_features"])

content_similarity = cosine_similarity(tfidf_matrix)

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



def recommend_cf(title, n=10):
    real_title = find_movie(title)

    if real_title is None:
        return ["Movie not found!"]

    movie_id = movies[movies["title"] == real_title]["movieId"].values[0]

    all_users = ratings["userId"].unique()
    predictions = []

    for user in all_users[:100]:

        pred = model.predict(user, movie_id)
        predictions.append((user, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_users = [u[0] for u in predictions[:n]]

    final_recs = []
    for user in top_users:
        user_data = ratings[ratings["userId"] == user]
        top_movie = user_data.sort_values("rating", ascending=False)["movieId"].head(1)

        for m in top_movie:
            title_name = movies[movies["movieId"] == m]["title"].values[0]
            final_recs.append(title_name)

    return final_recs

st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter movie name:")

if st.button("Recommend"):

    st.write("### ⭐ Content-Based Recommendations:")
    for m in recommend_content(movie_name):
        st.write("→", m)

    st.write("---")

    st.write("### ⭐ Collaborative Filtering Recommendations:")
    for m in recommend_cf(movie_name):
        st.write("→", m)
