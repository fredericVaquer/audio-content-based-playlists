import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import os

# --- 1. Load Tag Names ---
def get_discogs_tags():
    """Downloads the 400 style tags from the user-verified Essentia link."""
    # The URL you found is the metadata for the base feature extractor
    url = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json"
    
    tag_dir = "models"
    tag_file = os.path.join(tag_dir, "style_tags.json")

    if not os.path.exists(tag_dir):
        os.makedirs(tag_dir)

    # Download if missing or empty
    if not os.path.exists(tag_file) or os.path.getsize(tag_file) == 0:
        print(f"Downloading style tags from {url}...")
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            
            data = r.json()
            # The style names are inside the 'classes' key
            if 'classes' in data:
                tags = data['classes']
                with open(tag_file, 'w') as f:
                    json.dump(tags, f)
                print(f"Successfully saved {len(tags)} style tags.")
            else:
                print("Error: Could not find 'classes' key in the JSON.")
                return [f"Style_{i}" for i in range(400)] # Fallback
        except Exception as e:
            print(f"Failed to download style tags: {e}")
            return [f"Style_{i}" for i in range(400)] # Fallback

    # Load and return the names
    with open(tag_file, 'r') as f:
        return json.load(f)

# --- 2. Processing Logic ---
def process_analysis_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tags = get_discogs_tags()
    rows = []

    for path, feat in data.items():
        # Basic Meta
        row = {
            'file': os.path.basename(path),
            'tempo': feat.get('tempo'),
            'loudness': feat.get('loudness'),
            'danceability': feat.get('danceability')[0] if isinstance(feat.get('danceability'), list) else feat.get('danceability'),
            'voice': feat.get('voice_presence')[0] if isinstance(feat.get('voice_presence'), list) else feat.get('voice_presence'),
        }

        # Keys - Logic to extract key string from the nested dict
        for profile in ['temperley', 'krumhansl', 'edma']:
            p_data = feat.get('key_info', {}).get(profile, {})
            row[f'key_{profile}'] = f"{p_data.get('key')} {p_data.get('scale')}"

        # Styles
        # Get index of max activation
        style_probs = np.array(feat.get('music_styles', []))
        if len(style_probs) > 0:
            max_idx = np.argmax(style_probs)
            full_style = tags[max_idx] # e.g., "Electronic---Techno"
            row['primary_style'] = full_style
            row['parent_genre'] = full_style.split('---')[0]
        
        rows.append(row)
    
    return pd.DataFrame(rows), tags, data

# --- 3. Visualization and Reporting ---
def generate_report(json_path):
    df, tags, raw_data = process_analysis_results(json_path)
    
    # Setup Figure
    fig, axes = plt.subplots(3, 2, figsize=(20, 22))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Q1: Parent Genres
    genre_counts = df['parent_genre'].value_counts()
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=axes[0,0], palette="viridis")
    axes[0,0].set_title("Distribution of Parent Broad Genres")
    axes[0,0].set_xlabel("Track Count")

    # Q2: Tempo & Danceability
    sns.histplot(df['tempo'], bins=30, kde=True, ax=axes[0,1], color='teal')
    axes[0,1].set_title("Tempo Distribution (BPM)")
    
    sns.histplot(df['danceability'], bins=20, kde=True, ax=axes[1,0], color='magenta')
    axes[1,0].set_title("Danceability Distribution (0=Low, 1=High)")

    # Q3: Loudness
    sns.histplot(df['loudness'], bins=30, kde=True, ax=axes[1,1], color='orange')
    axes[1,1].set_title("Integrated Loudness (LUFS)")
    axes[1,1].axvline(-14, color='red', linestyle='--', label='Streaming Std (-14 LUFS)')
    axes[1,1].legend()

    # Q4: Key Profile Agreement (Temperley used for the plot)
    key_order = ["C major", "C# major", "D major", "D# major", "E major", "F major", 
                 "F# major", "G major", "G# major", "A major", "A# major", "B major",
                 "C minor", "C# minor", "D minor", "D# minor", "E minor", "F minor", 
                 "F# minor", "G minor", "G# minor", "A minor", "A# minor", "B minor"]
    
    # Filter only keys in our expected list for a cleaner plot
    top_keys = df['key_temperley'].value_counts().reindex(key_order).fillna(0)
    top_keys.plot(kind='bar', ax=axes[2,0], color='darkblue')
    axes[2,0].set_title("Tonality Distribution (Temperley Profile)")

    # Q5: Vocal vs Instrumental
    df['vocal_label'] = df['voice'].apply(lambda x: 'Vocal' if x > 0.5 else 'Instrumental')
    df['vocal_label'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[2,1], colors=['#66b3ff','#99ff99'])
    axes[2,1].set_title("Vocal vs Instrumental Ratio")

    plt.tight_layout()
    plt.savefig('music_collection_report.png')
    
    # --- STATISTICS FOR WRITTEN REPORT ---
    print("\n--- REPORT STATISTICS ---")
    
    # Key Agreement
    agreement = (df['key_temperley'] == df['key_krumhansl']) & (df['key_krumhansl'] == df['key_edma'])
    agree_pct = agreement.mean() * 100
    print(f"Key Agreement (All 3 Profiles): {agree_pct:.2f}%")

    # Style TSV Export
    # Create a detailed TSV of all styles for all tracks
    style_results = []
    for path, feat in raw_data.items():
        probs = feat.get('music_styles', [])
        for i, p in enumerate(probs):
            if p > 0.1: # Only export styles above 10% activation
                style_results.append({'file': os.path.basename(path), 'style': tags[i], 'activation': p})
    
    pd.DataFrame(style_results).to_csv('style_distribution_full.tsv', sep='\t', index=False)
    print("Full styles result exported to 'style_distribution_full.tsv'")
    plt.show()

if __name__ == "__main__":
    generate_report('analysis_results.json')