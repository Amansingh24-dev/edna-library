import streamlit as st
import pandas as pd
import requests
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

# -----------------------------
# OpenAI API Key
# -----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]  # store safely as secret

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Advanced AI eDNA Platform", layout="wide")
st.title("🌏 Advanced AI-Powered eDNA Analysis Platform")

# -----------------------------
# Sidebar - Multi Dataset Support
# -----------------------------
st.sidebar.header("Settings")
uploaded_files = st.sidebar.file_uploader(
    "Upload your CSV datasets", type=["csv"], accept_multiple_files=True
)
country = st.sidebar.text_input("Country code for GBIF API", "IN")
limit = st.sidebar.slider("Number of records (GBIF API)", 10, 500, 50)

# -----------------------------
# Load or Fetch Dataset
# -----------------------------
@st.cache_data
def fetch_gbif_data(country, limit):
    url = f"https://api.gbif.org/v1/occurrence/search?country={country}&limit={limit}"
    response = requests.get(url).json()
    data = response.get('results', [])
    df = pd.DataFrame(data)
    return df

dfs = []
if uploaded_files:
    for f in uploaded_files:
        dfs.append(pd.read_csv(f))
else:
    dfs.append(fetch_gbif_data(country, limit))

df = pd.concat(dfs, ignore_index=True)
st.subheader(f"Combined Dataset Preview ({len(df)} records)")
st.dataframe(df.head())

# -----------------------------
# Interactive Filters
# -----------------------------
st.sidebar.subheader("Filters")
if 'species' in df.columns:
    species_filter = st.sidebar.multiselect("Select species", df['species'].dropna().unique())
    if species_filter:
        df = df[df['species'].isin(species_filter)]
if 'year' in df.columns:
    year_filter = st.sidebar.slider("Year range", int(df['year'].min()), int(df['year'].max()), (int(df['year'].min()), int(df['year'].max())))
    df = df[df['year'].between(*year_filter)]

# -----------------------------
# PCA & Map
# -----------------------------
if 'decimalLatitude' in df.columns and 'decimalLongitude' in df.columns:
    coords = df[['decimalLatitude','decimalLongitude']].fillna(0)
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(coords), columns=['PC1','PC2'])
    
    st.subheader("PCA Plot of Locations")
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='PC1', y='PC2', data=df_pca)
    st.pyplot(plt)
    
    st.subheader("Interactive Map")
    fig = px.scatter_mapbox(df, lat='decimalLatitude', lon='decimalLongitude',
                            hover_name='species', zoom=1, height=400)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

# -----------------------------
# Biodiversity Metrics
# -----------------------------
def shannon_diversity(series):
    counts = series.value_counts()
    proportions = counts / counts.sum()
    return -sum(proportions * np.log(proportions))

def simpson_diversity(series):
    counts = series.value_counts()
    proportions = counts / counts.sum()
    return 1 - sum(proportions**2)

st.subheader("🌿 Biodiversity Metrics")
if 'species' in df.columns:
    richness = df['species'].nunique()
    shannon = shannon_diversity(df['species'])
    simpson = simpson_diversity(df['species'])
    st.write(f"Species Richness: {richness}")
    st.write(f"Shannon Diversity Index: {shannon:.3f}")
    st.write(f"Simpson Diversity Index: {simpson:.3f}")

# -----------------------------
# Temporal Trends
# -----------------------------
if 'year' in df.columns:
    st.subheader("📈 Temporal Trends of Species Occurrence")
    temporal = df.groupby('year')['species'].nunique()
    st.line_chart(temporal)

# -----------------------------
# Embeddings for AI Chatbot
# -----------------------------
@st.cache_resource
def embed_dataset(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = df.fillna('').astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings, texts

embeddings, texts = embed_dataset(df)

# -----------------------------
# Persistent AI Memory
# -----------------------------
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def ai_chat(question, embeddings, texts, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = model.encode([question])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(sims)[-top_k:][::-1]
    context = "\n".join([texts[i] for i in top_idx])
    
    prompt = f"""
    You are an intelligent eDNA assistant.
    Use ONLY the context below to answer questions.
    CONTEXT:
    {context}

    Previous chats: {st.session_state['chat_history']}

    QUESTION:
    {question}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user", "content": prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content
    st.session_state['chat_history'].append({"Q": question, "A": answer})
    return answer

st.subheader("💬 AI Chatbot for eDNA Analysis")
user_question = st.text_input("Ask the AI about your dataset")
if user_question:
    answer = ai_chat(user_question, embeddings, texts)
    st.write("**AI Answer:**", answer)

# -----------------------------
# Quick Analysis
# -----------------------------
st.subheader("🔍 Quick Analysis")
if st.button("Top 20 Species Count"):
    if 'species' in df.columns:
        st.bar_chart(df['species'].value_counts().head(20))
    else:
        st.write("No species column found.")

if st.button("Dataset Summary"):
    st.write(df.describe(include='all'))

# -----------------------------
# Downloadable Results
# -----------------------------
st.subheader("📥 Export Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Dataset CSV",
    data=csv,
    file_name='edna_analysis.csv',
    mime='text/csv',
)
