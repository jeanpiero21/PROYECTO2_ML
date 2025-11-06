# ============================================================
# 🎬 SISTEMA DE RECOMENDACIÓN VISUAL — STREAMLIT APP (API TMDb vía imdbId)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.metrics import pairwise_distances

# ============================================================
# ⚙️ CONFIGURACIÓN GLOBAL
# ============================================================
API_KEY = "554915d1c4f36cd57e9b7d6790339448"  # ⚠️ Tu clave TMDb
META_TRAIN = "train_metadata_with_clusters.csv"
META_TEST = "test_metadata_with_clusters.csv"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# ============================================================
# 🧩 FUNCIONES AUXILIARES
# ============================================================

@st.cache_data
def load_csv_metadata(path: str):
    """Lee CSV con metadata + embeddings + cluster."""
    df = pd.read_csv(path)
    df["imdbId"] = df["imdbId"].astype(str).str.zfill(7)
    return df


@st.cache_data
def get_poster_url_from_imdb(imdb_id: str):
    """
    Consulta la API de TMDb usando imdbId para obtener la URL del póster.
    Ejemplo de endpoint: https://api.themoviedb.org/3/find/tt0012349?api_key=KEY&external_source=imdb_id
    """
    if not imdb_id or pd.isna(imdb_id):
        return None
    try:
        imdb_formatted = f"tt{imdb_id}"
        url = f"https://api.themoviedb.org/3/find/{imdb_formatted}?api_key={API_KEY}&external_source=imdb_id"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("movie_results", [])
            if results:
                poster_path = results[0].get("poster_path")
                if poster_path:
                    return TMDB_IMAGE_BASE + poster_path
    except Exception:
        pass
    return None


@st.cache_data
def load_all_data():
    """Carga los CSV de train y test + matrices de embeddings."""
    df_train = load_csv_metadata(META_TRAIN)
    df_test = load_csv_metadata(META_TEST)

    emb_cols = [c for c in df_train.columns if c.lower().startswith("umap") or c.lower().startswith("pca")]
    if not emb_cols:
        st.error("No se encontraron columnas UMAP/PCA en los CSVs.")
        st.stop()

    X_train = df_train[emb_cols].values
    X_test = df_test[emb_cols].values
    return df_train, df_test, X_train, X_test, emb_cols


# ============================================================
# 🖥️ INTERFAZ PRINCIPAL
# ============================================================
st.set_page_config(page_title="🎥 Recomendador (API TMDb + Clúster + Género)", layout="wide")

with st.spinner("Cargando CSVs preprocesados..."):
    df_train, df_test, X_train, X_test, emb_cols = load_all_data()

st.success(f"✅ Cargados {len(df_train)} películas de train y {len(df_test)} de test.")


# ============================================================
# 🧭 SIDEBAR
# ============================================================
st.sidebar.title("🎬 MENÚ")
modo = st.sidebar.radio("Sección:", ["Visualizador (TEST)", "Recomendaciones (TRAIN)"])

# Filtro de género textual
all_genres = sorted(set("|".join(df_train["Genre"].dropna()).split("|")))
filtro_genero = st.sidebar.selectbox("🎭 Filtrar por género (dentro del cluster)", ["Todos"] + all_genres)

if st.sidebar.button("🔄 Refrescar"):
    st.rerun()


# ============================================================
# 🎞️ VISUALIZADOR (TEST)
# ============================================================
if modo == "Visualizador (TEST)":
    st.title("🎥 Visualizador — Conjunto TEST")

    df_show = df_test.copy()
    muestras = df_show.sample(min(20, len(df_show)), random_state=42)

    cols = st.columns(5)
    for i, (_, row) in enumerate(muestras.iterrows()):
        imdb_id = row["imdbId"]
        poster_url = get_poster_url_from_imdb(imdb_id)

        with cols[i % 5]:
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.image(np.zeros((200, 150, 3)), caption="Sin imagen")
            st.caption(f"🎬 {row['Title']}\n⭐ {row.get('rating','-')}\n🎭 {row.get('Genre','-')}")
            if st.button(f"🔍 Recomendar {imdb_id}", key=f"rec_{imdb_id}"):
                st.session_state["pelicula_base_test"] = imdb_id
                st.session_state["modo"] = "Recomendaciones (TRAIN)"
                st.rerun()


# ============================================================
# 🎯 RECOMENDACIONES (TRAIN)
# ============================================================
else:
    st.title("🎯 Recomendaciones — corpus TRAIN (API TMDb + Cluster + Género textual)")

    if "pelicula_base_test" not in st.session_state:
        st.warning("Primero selecciona una película desde el Visualizador (TEST).")
        st.stop()

    base_id = st.session_state["pelicula_base_test"]
    df_base = df_test[df_test["imdbId"] == base_id]

    if df_base.empty:
        st.error("No se encontró la película base en TEST.")
        st.stop()

    base_vec = df_base[emb_cols].values[0]
    base_info = df_base.iloc[0]
    base_poster = get_poster_url_from_imdb(base_info["imdbId"])

    st.sidebar.image(base_poster if base_poster else np.zeros((200,150,3)), use_container_width=True)
    st.sidebar.markdown(f"**🎬 {base_info['Title']}**")
    st.sidebar.markdown(f"⭐ Rating: {base_info.get('rating','-')}")
    st.sidebar.markdown(f"🎭 Género: {base_info.get('Genre','-')}")
    if st.sidebar.button("⬅️ Volver al Visualizador"):
        st.session_state["modo"] = "Visualizador (TEST)"
        st.rerun()

    metrica = st.selectbox("📏 Métrica de similitud:", ["euclidean", "manhattan", "cosine"])
    top_n = st.slider("Número de recomendaciones:", 4, 20, 8)

    # --- Cluster base ---
    dist_all = pairwise_distances(base_vec.reshape(1, -1), df_train[emb_cols].values, metric=metrica).flatten()
    idx_nearest = np.argmin(dist_all)
    cluster_base = df_train.iloc[idx_nearest]["Cluster"]
    st.info(f"📦 Cluster asignado automáticamente: **{int(cluster_base)}**")

    df_cluster = df_train[df_train["Cluster"] == cluster_base].copy()
    if df_cluster.empty:
        st.warning("No hay películas en este cluster.")
        st.stop()

    X_cluster = df_cluster[emb_cols].values
    dist_cluster = pairwise_distances(base_vec.reshape(1, -1), X_cluster, metric=metrica).flatten()
    df_cluster["Distancia"] = dist_cluster
    df_cluster = df_cluster.sort_values("Distancia", ascending=True).reset_index(drop=True)

    # Filtro de género
    if filtro_genero != "Todos":
        df_cluster = df_cluster[
            df_cluster["Genre"].fillna("").apply(
                lambda g: filtro_genero.lower() in [x.strip().lower() for x in g.split("|")]
            )
        ]

    if df_cluster.empty:
        st.warning(f"No hay películas del cluster {int(cluster_base)} con el género '{filtro_genero}'.")
        st.stop()

    df_cluster = df_cluster.head(top_n)
    st.subheader(f"🎬 Películas similares a: {base_info['Title']} (Cluster {int(cluster_base)})")

    cols = st.columns(4)
    for i, (_, row) in enumerate(df_cluster.iterrows()):
        poster_url = get_poster_url_from_imdb(row["imdbId"])
        with cols[i % 4]:
            if poster_url:
                st.image(poster_url, use_container_width=True)
            else:
                st.image(np.zeros((200,150,3)), caption="Sin imagen")
            st.caption(
                f"🎬 {row['Title']}\n"
                f"⭐ {row.get('rating','-')}\n"
                f"🎭 {row.get('Genre','-')}\n"
                f"📦 Cluster: {int(row['Cluster'])}\n"
                f"📏 Distancia ({metrica}): {row['Distancia']:.4f}"
            )
