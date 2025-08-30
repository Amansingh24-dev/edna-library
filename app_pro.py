import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan
import faiss
import json
import base64
import tempfile

# Optional GPT chatbot
from openai import OpenAI
import os

st.set_page_config(page_title="🌊 eDNA Pro + AI Chatbot", layout="wide")
st.title("🌊 eDNA Pro — Worldwide Dataset + AI Chatbot")

# -------------------------
# Utilities
# -------------------------
def read_fasta_like(data: str):
    lines = [l.strip() for l in data.splitlines() if l.strip() != ""]
    seqs = []
    cur = []
    for l in lines:
        if l.startswith(">") or l.startswith("@"):  # FASTA / FASTQ header
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(l)
    if cur:
        seqs.append("".join(cur))
    return seqs


def kmer_count_matrix(sequences, k=4, max_features=2000):
    vec = CountVectorizer(analyzer='char', ngram_range=(k, k), max_features=max_features)
    X = vec.fit_transform(sequences)
    return X.toarray(), vec.get_feature_names_out()


def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype(np.float32))
    return index

# -------------------------
# File upload / paste
# -------------------------
uploaded = st.file_uploader("Upload FASTA/FASTQ or .csv", type=["fasta","fa","fastq","txt","csv"])
paste = st.text_area("Or paste raw sequences here (FASTA-like) — headers allowed", height=120)

raw = None
if uploaded is not None:
    raw = uploaded.read().decode('utf-8')
elif paste and len(paste.strip()) > 0:
    raw = paste

use_worldwide = st.checkbox("Include worldwide reference dataset (demo) ")

sequences = []
if raw:
    sequences = read_fasta_like(raw)
    sequences = [s.upper().replace("N","") for s in sequences if len(s) >= 20]
    st.success(f"Loaded {len(sequences)} sequences")

    show_limit = st.slider("Show first N sequences", 5, min(50, max(5,len(sequences))), 10)
    st.write(sequences[:show_limit])

if sequences:
    k = st.number_input("k-mer size", min_value=3, max_value=7, value=4)
    X, kmer_names = kmer_count_matrix(sequences, k=k)
    st.write("Feature matrix shape:", X.shape)

    Xs = StandardScaler(with_mean=False).fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    # Clustering
    st.subheader("Unsupervised clustering (HDBSCAN)")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(Xs)
    uniq, counts = np.unique(labels, return_counts=True)
    df_counts = pd.DataFrame({"cluster": uniq, "count": counts})
    st.write(df_counts)

    fig, ax = plt.subplots()
    scatter = ax.scatter(Xp[:,0], Xp[:,1], c=labels, cmap='tab10', s=40)
    ax.set_title("PCA projection of sequences")
    st.pyplot(fig)

    # Biodiversity
    st.subheader("Biodiversity metrics")
    counts_series = pd.Series(labels).value_counts()
    richness = (counts_series.index != -1).sum()
    proportions = (counts_series[counts_series.index!=-1] / counts_series.sum())
    shannon = -np.sum(proportions * np.log(proportions+1e-12))
    st.metric("Estimated richness (clusters)", richness)
    st.metric("Shannon index (approx)", f"{shannon:.3f}")

    # Simulated Taxonomy or worldwide
    st.subheader("Taxonomy Assignment")
    fake_taxa = ["Protist","Cnidarian","Metazoan","Fungi","Other"]
    rng = np.random.default_rng(42)
    taxa = rng.choice(fake_taxa, size=len(sequences))

    if use_worldwide:
        # Demo: add a few worldwide sequences to show retrieval
        worldwide_sequences = ["ATGCGTACGTTAGC","CGTACGATCGTACG","GCTAGCTAGCATG"]
        sequences += worldwide_sequences
        taxa = np.concatenate([taxa, ["Worldwide1","Worldwide2","Worldwide3"]])
        X, kmer_names = kmer_count_matrix(sequences, k=k)
        Xs = StandardScaler(with_mean=False).fit_transform(X)

    df = pd.DataFrame({"sequence": [s[:80]+"..." for s in sequences], "cluster": labels, "predicted_taxon": taxa})
    st.dataframe(df.head(200))

    # Build FAISS
    st.subheader("FAISS retrieval index")
    emb = Xs.astype(np.float32)
    index = build_faiss_index(emb)
    corpus_texts = [f"SequenceID:{i} | len:{len(seq)} | cluster:{labels[i]} | taxon:{taxa[i]} | seq:{seq[:200]}" for i, seq in enumerate(sequences)]

    st.session_state['corpus'] = corpus_texts
    st.session_state['faiss_index'] = index
    st.session_state['embeddings'] = emb

    # -------------------------
    # Chatbot
    # -------------------------
    st.subheader("AI Chatbot — Ask about your dataset")
    query = st.text_input("Ask a question (e.g., 'which clusters look novel?')")

    if 'faiss_index' in st.session_state and query:
        qvec, _ = kmer_count_matrix([query.upper()], k=k, max_features=len(kmer_names))
        q = np.zeros((1, emb.shape[1]), dtype=np.float32)
        q[:, :qvec.shape[1]] = qvec.astype(np.float32)
        D, I = st.session_state['faiss_index'].search(q, k=5)
        hits = [st.session_state['corpus'][idx] for idx in I[0]]
        st.write("Top matches:")
        for h in hits:
            st.write(h)

        # Optional GPT answer (requires OPENAI_API_KEY in environment)
        if os.getenv("OPENAI_API_KEY"):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = f"You are an expert eDNA analyst. The user query is: {query}. Here are the top 5 sequences retrieved: {hits}. Provide a summary answer in plain English." 
            try:
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role":"user","content":prompt}])
                answer = response.choices[0].message.content
                st.info(answer)
            except Exception as e:
                st.warning(f"GPT query failed: {e}")

    # Export report
    if st.button("Export report (JSON)"):
        report = {"n_sequences": len(sequences), "clusters": df_counts.to_dict(orient='records'), "shannon": float(shannon), "taxa_preview": taxa.tolist()[:50]}
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        fp.write(json.dumps(report, indent=2).encode('utf-8'))
        fp.close()
        with open(fp.name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="edna_report.json">Download report</a>'
            st.markdown(href, unsafe_allow_html=True)

else:
    st.info("Upload sequences to start analysis.")
