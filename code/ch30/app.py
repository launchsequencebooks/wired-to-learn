# The organizer's user interface (run: streamlit run app.py).
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
from organizer import Organizer

st.set_page_config(page_title="Smart Photo Organizer", layout="wide")
st.title("Smart Photo Organizer")
folder = Path(st.sidebar.text_input("Photo folder", "photos"))


@st.cache_resource                                           # load the networks once per server
def organize(folder):
    files = sorted(folder.glob("*.png"))
    if not files:
        st.error(f"No PNG files in {folder}. Run make_demo_photos.py first."); st.stop()
    images = np.stack([np.asarray(Image.open(f).convert("L").resize((28, 28)), dtype=np.float32) / 255
                       for f in files])[:, None]
    return Organizer().ingest(images, [f.name for f in files]), files


org, files = organize(folder)
path = {f.name: f for f in files}
thumb = lambda name: Image.open(path[name]).resize((56, 56), Image.NEAREST)
albums = org.albums()
st.sidebar.metric("Photos", len(files))
st.sidebar.metric("Filed automatically", f"{1 - (len(albums.get('Review', [])) + len(albums.get('Not clothing?', []))) / len(files):.0%}")

tab_albums, tab_similar = st.tabs(["Albums", "Find similar"])
with tab_albums:
    order = sorted(albums, key=lambda k: (k not in ("Not clothing?", "Review"), -len(albums[k])))   # needs a person: first
    for name in order:
        st.markdown(f"**{name}** ({len(albums[name])})")
        st.image([thumb(n) for n in albums[name][:11]], width=56)
with tab_similar:
    query = st.selectbox("Photo", org.names)
    st.image(thumb(query), width=112, caption=f"{org.tags[org.names.index(query)]}")
    hits = org.similar(query, k=6)
    st.image([thumb(n) for n, _ in hits], width=84, caption=[f"{org.tags[org.names.index(n)]} {s:.2f}" for n, s in hits])
