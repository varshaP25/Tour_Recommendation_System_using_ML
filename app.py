
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("Top Indian Places to Visit.csv")
    df.fillna('', inplace=True)
    df['combined_features'] = (
        df['City'].astype(str) + ' ' +
        df['Type'].astype(str) + ' ' +
        df['Significance'].astype(str)
    )
    return df

df = load_data()

# Vectorize and compute similarity
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df['combined_features'])

# ------------------- Streamlit UI ---------------------
st.title("🧭 Indian Tour Recommendation System")

tab1, tab2 = st.tabs(["🔎 Search by Place", "📝 Keyword Search"])

# ------------- Tab 1: Traditional Place Recommendation -------------
with tab1:
    st.subheader("Find similar places to a location you like")

    place = st.selectbox("Select a place:", sorted(df['Name'].unique()))

    with st.expander("🔍 Optional Filters"):
        state = st.selectbox("Filter by State", [""] + sorted(df['State'].unique()))
        rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0, 0.1)
        place_type = st.selectbox("Place Type", [""] + sorted(df['Type'].unique()))
        season = st.selectbox("Best Time to Visit", [""] + sorted(df['Best Time to visit'].unique()))

    state = state if state else None
    place_type = place_type if place_type else None
    season = season if season else None
    min_rating = rating if rating > 0 else None

    if st.button("🔁 Recommend Similar Places", key="place_button"):
        if place not in df['Name'].values:
            st.warning("Selected place not found.")
        else:
            idx = df[df['Name'] == place].index[0]
            similarity = cosine_similarity(feature_matrix)
            sim_scores = list(enumerate(similarity[idx]))
            sim_scores = sim_scores[1:]
            sim_scores = sorted(sim_scores, key=lambda x: (x[1], df.iloc[x[0]]['Google review rating']), reverse=True)

            recommendations = []
            for i in sim_scores:
                row = df.iloc[i[0]]
                if state and row['State'].strip().lower() != state.strip().lower():
                    continue
                if min_rating and row['Google review rating'] < min_rating:
                    continue
                if place_type and place_type.lower() not in row['Type'].lower():
                    continue
                if season and season.lower() not in row['Best Time to visit'].lower():
                    continue
                recommendations.append(row['Name'])
                if len(recommendations) == 5:
                    break

            st.subheader("✨ You might also like:")
            if recommendations:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.info("No matching places found with selected filters.")

# ------------- Tab 2: Keyword-Based Recommendation -------------
with tab2:
    st.subheader("Get recommendations based on any keyword")

    keyword_input = st.text_input("Enter any keyword (e.g., beach, temple, trekking in Kerala)")

    if st.button("🔍 Search by Keyword"):
        if keyword_input.strip() == "":
            st.warning("Please enter a keyword.")
        else:
            query_vec = vectorizer.transform([keyword_input])
            sim_scores = cosine_similarity(query_vec, feature_matrix).flatten()
            ranked_indices = sim_scores.argsort()[::-1]

            st.subheader("✨ Recommended Places Based on Keyword:")
            found = False
            for idx in ranked_indices[:10]:
                if sim_scores[idx] > 0:
                    st.markdown(f"- {df.iloc[idx]['Name']} ({df.iloc[idx]['State']})")
                    found = True
            if not found:
                st.info("No matching places found for the keyword.")
