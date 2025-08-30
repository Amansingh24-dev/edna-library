# app_pro.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
import faiss
import tempfile, os, json
import base64

st.set_page_config(page_title="eDNA Pro Prototype", layout="wide")

st.title("🌊 eDNA Pro Prototype — AI + Retrieval Chatbot")
st.markdown("Upload FASTA/FASTQ or paste sequences. This demo shows preprocessing, embeddings (k-mer), clustering, diversity and a small retrieval chatbot.")

# -------------------------
# Utilities
# -------------------------
def read_fasta_like(data: str):
    lines = [l.strip() for l in data.splitlines() if l.strip()!=""]
    seqs = []
    cur = []
    for l in lines:
        if l.startswith(">") or l.startswith("@"):
            if cur:
                seqs.append("".join(cur))
                cur=[]
        else:
            cur.append(l)
    if cur:
        seqs.append("".join(cur))
    return seqs

def kmer_count_matrix(sequences, k=4, max_features=2000):
    vec = CountVectorizer(analyzer='char', ngram_range=(k,k), max_features=max_features)
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
uploaded = st.file_uploader("Upload FASTA/FASTQ (.fasta .fa .fastq .txt) or .csv", type=["fasta","fa","fastq","txt","csv"])
paste = st.text_area("Or paste raw sequences here (FASTA-like) — headers allowed", height=120)

raw = None
if uploaded is not None:
    raw = uploaded.read().decode('utf-8')
elif paste and len(paste.strip())>0:
    raw = paste

if raw:
    sequences = read_fasta_like(raw)
    sequences = [s.upper().replace("N","") for s in sequences if len(s)>=20]
    st.success(f"Loaded {len(sequences)} sequences (trimmed empty/N-only)")
    show_limit = st.slider("Show first N sequences", 5, min(50, max(5,len(sequences))), 10)
    st.write(sequences[:show_limit])

    # k-mer features
    k = st.number_input("k-mer size", min_value=3, max_value=7, value=4)
    X, kmer_names = kmer_count_matrix(sequences, k=k)
    st.write("Feature matrix shape:", X.shape)

    # scale & PCA
    Xs = StandardScaler(with_mean=False).fit_transform(X)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    # clustering
    st.subheader("Unsupervised clustering (HDBSCAN)")
    clusterer = HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(Xs)
    uniq, counts = np.unique(labels, return_counts=True)
    df_counts = pd.DataFrame({"cluster":uniq, "count":counts})
    st.write(df_counts)

    fig, ax = plt.subplots()
    scatter = ax.scatter(Xp[:,0], Xp[:,1], c=labels, cmap='tab10', s=40)
    ax.set_title("PCA projection of sequences")
    st.pyplot(fig)

    # biodiversity
    st.subheader("Biodiversity metrics")
    # map negative label (-1) = noise; exclude noise for richness
    assigned = labels.copy()
    counts_series = pd.Series(assigned).value_counts()
    richness = (counts_series.index != -1).sum()
    proportions = (counts_series[counts_series.index!=-1] / counts_series.sum())
    shannon = -np.sum(proportions * np.log(proportions+1e-12))
    st.metric("Estimated richness (clusters)", richness)
    st.metric("Shannon index (approx)", f"{shannon:.3f}")

    # Simulated taxonomy assign (toy; replace with real classifier later)
    st.subheader("Simulated Taxonomy (toy)")
    fake_taxa = ["Protist","Cnidarian","Metazoan","Fungi","Other"]
    rng = np.random.default_rng(42)
    taxa = rng.choice(fake_taxa, size=len(sequences))
    df = pd.DataFrame({"sequence": [s[:80]+"..." for s in sequences], "cluster":labels, "predicted_taxon":taxa})
    st.dataframe(df.head(200))

    # build FAISS index for sequences (k-mer vectors as embeddings)
    st.subheader("Build retrieval index for chatbot")
    emb = Xs.astype(np.float32)
    index = build_faiss_index(emb)
    st.success("FAISS index built (k-mer embeddings)")

    # save small "corpus" for retrieval
    corpus_texts = []
    for i, seq in enumerate(sequences):
        corpus_texts.append(f"SequenceID:{i} | len:{len(seq)} | cluster:{labels[i]} | taxon:{taxa[i]} | seq:{seq[:200]}")

    # store corpus in session state
    st.session_state['corpus'] = corpus_texts
    st.session_state['faiss_index'] = index
    st.session_state['embeddings'] = emb

    # -------------------------
    # Chatbot: retrieval + simple answer
    # -------------------------
    st.subheader("Chatbot (retrieval-based) — ask about your data")
    query = st.text_input("Ask a question about the dataset (e.g., 'which clusters look novel?')")
    if 'faiss_index' in st.session_state and query:
        # naive: convert query to k-mer counts by treating query as text (toy)
        qvec, _ = kmer_count_matrix([query.upper()], k=k, max_features=len(kmer_names))
        # pad/truncate qvec to match shape
        q = np.zeros((1, emb.shape[1]), dtype=np.float32)
        q[:, :qvec.shape[1]] = qvec.astype(np.float32)
        D, I = st.session_state['faiss_index'].search(q, k=5)
        hits = []
        for idx in I[0]:
            hits.append(st.session_state['corpus'][idx])
        st.write("Top matches (toy retrieval):")
        for h in hits:
            st.write(h)
        # basic canned answer
        if any("cluster:-1" in h for h in hits):
            st.info("There are noise sequences (cluster -1) which may be low-quality or novel.")
        else:
            st.success("Top matches show sequences assigned to known clusters; consider further alignment for species-level ID.")

    # Export report
    if st.button("Export report (JSON)"):
        report = {
            "n_sequences": len(sequences),
            "clusters": df_counts.to_dict(orient='records'),
            "shannon": float(shannon),
            "taxa_preview": taxa.tolist()[:50]
        }
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        fp.write(json.dumps(report, indent=2).encode('utf-8'))
        fp.close()
        with open(fp.name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="edna_report.json">Download report</a>'
            st.markdown(href, unsafe_allow_html=True)
else:
    st.info("Upload FASTA/FASTQ or paste sequences to start. For best demo, paste several (>20) sequences.")
