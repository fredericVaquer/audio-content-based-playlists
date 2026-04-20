import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import torch
from transformers import AutoProcessor, ClapModel
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. CONFIG & DATA LOADING ---
st.set_page_config(page_title="Audio Content Discovery", layout="wide")

@st.cache_resource
def load_models():
    """Load CLAP once for text encoding"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "laion/larger_clap_music_and_speech"
    model = ClapModel.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, device

@st.cache_data
def load_analysis_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Flatten JSON for easy Pandas filtering
    rows = []
    for path, feat in data.items():
        row = {
            'path': path,
            'filename': os.path.basename(path),
            'tempo': feat.get('tempo'),
            'loudness': feat.get('loudness'),
            'danceability': feat.get('danceability')[0] if isinstance(feat.get('danceability'), list) else feat.get('danceability'),
            'voice': feat.get('voice_presence')[0] if isinstance(feat.get('voice_presence'), list) else feat.get('voice_presence'),
            # Using Temperley as requested
            'key': feat.get('key_info', {}).get('temperley', {}).get('key', 'Unknown'),
            'scale': feat.get('key_info', {}).get('temperley', {}).get('scale', 'Unknown'),
            'effnet_vec': feat.get('effnet_indices'),
            'clap_vec': feat.get('clap_indices'),
            'styles': feat.get('music_styles', [])
        }
        rows.append(row)
    return pd.DataFrame(rows), data

# Initialize
df, raw_data = load_analysis_data('analysis_results.json')
model, processor, device = load_models()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("Playlist Generator")
mode = st.sidebar.selectbox("Choose Mode", 
    ["Descriptor Queries", "Track Similarity", "Freeform Text Search"])

# --- 3. MODE 1: DESCRIPTOR QUERIES ---
if mode == "Descriptor Queries":
    st.header("🔍 Filter by Audio Descriptors")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        tempo_range = st.slider("Tempo (BPM)", 40, 220, (80, 140))
        voice_opt = st.radio("Voice presence", ["All", "Only Vocal", "Only Instrumental"])
    
    with col2:
        dance_range = st.slider("Danceability", 0.0, 1.0, (0.0, 1.0))
        key_select = st.selectbox("Key", ["All"] + sorted(df['key'].unique().tolist()))
        
    with col3:
        scale_select = st.selectbox("Scale", ["All", "major", "minor"])
        # Add Style search if you have the tag list...
        
    # Filtering Logic
    filtered = df[
        (df['tempo'] >= tempo_range[0]) & (df['tempo'] <= tempo_range[1]) &
        (df['danceability'] >= dance_range[0]) & (df['danceability'] <= dance_range[1])
    ]
    
    if voice_opt == "Only Vocal": filtered = filtered[filtered['voice'] > 0.5]
    if voice_opt == "Only Instrumental": filtered = filtered[filtered['voice'] <= 0.5]
    if key_select != "All": filtered = filtered[filtered['key'] == key_select]
    if scale_select != "All": filtered = filtered[filtered['scale'] == scale_select]

    st.write(f"Found **{len(filtered)}** tracks.")
    
    # Display top 10
    for idx, row in filtered.head(10).iterrows():
        with st.expander(f"🎵 {row['filename']} | {row['tempo']} BPM | {row['key']} {row['scale']}"):
            st.audio(row['path'])

# --- 4. MODE 2: TRACK SIMILARITY ---
elif mode == "Track Similarity":
    st.header("🤝 Similar Track Discovery")
    
    query_file = st.selectbox("Select a query track", df['filename'].tolist())
    query_row = df[df['filename'] == query_file].iloc[0]
    
    st.write("### Query Track")
    st.audio(query_row['path'])
    
    col_eff, col_clap = st.columns(2)
    
    # Helper for similarity
    def get_top_k(query_vec, vec_column, k=11):
        # Reshape for sklearn
        matrix = np.stack(df[vec_column].values)
        q = np.array(query_vec).reshape(1, -1)
        sims = cosine_similarity(q, matrix)[0]
        # Get indices of top k (excluding the query track itself)
        indices = np.argsort(sims)[::-1][1:k]
        return df.iloc[indices]

    with col_eff:
        st.subheader("Based on Effnet (Musical Vibe)")
        results_eff = get_top_k(query_row['effnet_vec'], 'effnet_vec')
        for _, r in results_eff.iterrows():
            st.write(r['filename'])
            st.audio(r['path'])

    with col_clap:
        st.subheader("Based on CLAP (Audio Semantic)")
        results_clap = get_top_k(query_row['clap_vec'], 'clap_vec')
        for _, r in results_clap.iterrows():
            st.write(r['filename'])
            st.audio(r['path'])

# --- 5. MODE 3: TEXT-AUDIO SEARCH ---
elif mode == "Freeform Text Search":
    st.header("⌨️ Text-to-Audio Search")
    query_text = st.text_input("Describe the sound you want (e.g., 'Heavy drums with a melancholic piano')")
    
    if query_text:
        # 1. Encode Text Query
        inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs).pooler_output
            # Ensure we have a tensor, then move to CPU
            if not torch.is_tensor(outputs):
                text_embed = getattr(outputs, "text_features", outputs[0])
            else:
                text_embed = outputs
            
            text_embed = text_embed.cpu().numpy()

        # 2. THE FIX: Force 2D shapes and handle the 65536 mismatch
        if text_embed.ndim > 2:
            text_embed = np.squeeze(text_embed)
        if text_embed.ndim == 1:
            text_embed = text_embed.reshape(1, -1)

        audio_matrix = np.stack(df['clap_vec'].values)
        
        # Check if audio is the wrong dimension (65536)
        if audio_matrix.shape[1] == 65536:
            st.warning("⚠️ Audio embeddings are unprojected (65536 dims). Attempting to pool...")
            # Reshape to (Tracks, 128, 512) and average across the sequence
            # (Note: This assumes 512 is the base dimension; change to 768 if needed)
            audio_matrix = audio_matrix.reshape(len(df), -1, 512).mean(axis=1)

        # Ensure both are exactly the same width now
        if text_embed.shape[1] != audio_matrix.shape[1]:
            st.error(f"Dimension Mismatch: Text is {text_embed.shape[1]}, Audio is {audio_matrix.shape[1]}. You must re-run analysis with 'get_audio_features'.")
            st.stop()
        
        # 3. Calculate Similarity
        sims = cosine_similarity(text_embed, audio_matrix)[0]
        
        top_indices = np.argsort(sims)[::-1][:10]
        
        st.write("### Results")
        for idx in top_indices:
            r = df.iloc[idx]
            st.write(f"Score: {sims[idx]:.4f} | {r['filename']}")
            st.audio(r['path'])

# --- 6. M3U8 EXPORT (Footer) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Export Results")

# 1. Determine which data to export based on your current tab/mode
if mode == "Descriptor Queries":
    export_df = filtered # From your filtering logic
elif mode == "Track Similarity":
    # Combine results from both models for the playlist
    export_df = pd.concat([results_eff, results_clap])
else: # Text Search
    # Top 10 results from the text search
    if 'top_indices' in locals():
        export_df = df.iloc[top_indices]
    else:
        export_df = pd.DataFrame()

# 2. Generate the M3U8 content
if not export_df.empty:
    m3u_lines = ["#EXTM3U"]
    for _, row in export_df.iterrows():
        # Add a comment with the song name and then the absolute path
        m3u_lines.append(f"#EXTINF:-1,{row['filename']}")
        m3u_lines.append(os.path.abspath(row['path']))
    
    m3u_content = "\n".join(m3u_lines)

    # 3. Use the download button directly (no nested 'if st.button')
    st.sidebar.download_button(
        label="📥 Download M3U8 Playlist",
        data=m3u_content,
        file_name="playlist.m3u8",
        mime="text/plain",
        help="Click to save the current results as a playlist file for VLC/Strawberry"
    )
else:
    st.sidebar.info("No tracks found to export.")