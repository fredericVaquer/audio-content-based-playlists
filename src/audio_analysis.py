import os
import sys
import json
import numpy as np
from tqdm import tqdm

# Silence framework noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import essentia.standard as es
import essentia
import torch
from transformers import AutoProcessor, ClapModel

essentia.log.infoActive = False
essentia.log.warningActive = False

# --- 1. Unit Functions ---

def extract_tempo(input_data, extractor):
    bpm, _, _, _, _ = extractor(input_data)
    return float(bpm)

def extract_keys(input_data, extractors_dict):
    """
    input_data: mono audio
    extractors_dict: {'temperley': extractor, 'krumhansl': extractor, 'edma': extractor}
    """
    results = {}
    for profile, extractor in extractors_dict.items():
        key, scale, strength = extractor(input_data)
        results[profile] = {
            "key": key,
            "scale": scale,
            "strength": float(strength)
        }
    return results

def extract_loudness(input_data, extractor):
    _, _, loudness, _ = extractor(input_data)
    return float(loudness)

def run_classifier(embeddings, extractor):
    # Workflow: embedding frames -> classifier output frames -> Averaging
    predictions = extractor(embeddings)
    mean_predictions = np.mean(predictions, axis=0)
    return mean_predictions.tolist()

def extract_effnet_embeddings(input_data, extractor):
    # Workflow: audio -> embedding frames -> Averaging
    embeddings = extractor(input_data)
    mean_embeddings = np.mean(embeddings, axis=0)
    return mean_embeddings, embeddings # Return both mean for storage and full for classifiers

def extract_TA_embeddings(input_data, extractor_tuple):
    model, processor, device = extractor_tuple
    inputs = processor(audio=input_data, return_tensors="pt", sampling_rate=48000, padding=True).to(device)
    with torch.no_grad():
        outputs = model.get_audio_features(**inputs)
    return outputs.pooler_output.cpu().numpy().flatten().tolist()

# --- 2. The Optimized Parser ---

def parse_folder(folder_path, output_file='analysis_results.json'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize State (Resume Capability)
    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                results = json.load(f)
                print(f"Resuming analysis. {len(results)} files already processed.")
            except json.JSONDecodeError:
                results = {}

    # 2. Instantiate Algorithms ONCE
    print("Loading models...")
    clap_model_id = "laion/larger_clap_music_and_speech"
    clap_model = ClapModel.from_pretrained(clap_model_id).to(device)
    clap_processor = AutoProcessor.from_pretrained(clap_model_id)

    extraction_registry = {
        'tempo': (extract_tempo, es.RhythmExtractor2013()),
        'key': (extract_keys, {
            'temperley': es.KeyExtractor(profileType='temperley'),
            'krumhansl': es.KeyExtractor(profileType='krumhansl'),
            'edma': es.KeyExtractor(profileType='edma')
        }),
        'loudness': (extract_loudness, es.LoudnessEBUR128()),
        'effnet': (extract_effnet_embeddings, es.TensorflowPredictEffnetDiscogs(
            graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")),
        'music_styles': (run_classifier, es.TensorflowPredict2D(
            graphFilename="models/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")),
        'voice_presence': (run_classifier, es.TensorflowPredict2D(
            graphFilename="models/voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")),
        'danceability': (run_classifier, es.TensorflowPredict2D(
            graphFilename="models/danceability-discogs-effnet-1.pb", output="model/Softmax")),
        'TA_embeddings': (extract_TA_embeddings, (clap_model, clap_processor, device))
    }

    # 3. Collect all MP3 files
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.mp3'):
                all_files.append(os.path.join(root, file))

    # 4. Process with Progress Bar
    for path in tqdm(all_files, desc="Analyzing Audio"):
        if path in results:
            continue  # Skip already analyzed files

        try:
            # Load audio once
            audio, sr, nbChannels, _, _, _ = es.AudioLoader(filename=path)()
            mono_audio = es.MonoMixer()(audio, nbChannels)

            track_data = {}
            
            # Pre-resample once per track
            audio_16k = es.Resample(inputSampleRate=sr, outputSampleRate=16000)(mono_audio)
            audio_48k = es.Resample(inputSampleRate=sr, outputSampleRate=48000)(mono_audio)

            # Execution Logic
            # Note: Effnet must run before classifiers
            mean_effnet, full_effnet = extraction_registry['effnet'][0](audio_16k, extraction_registry['effnet'][1])

            track_data['tempo'] = extraction_registry['tempo'][0](mono_audio, extraction_registry['tempo'][1])
            track_data['key_info'] = extraction_registry['key'][0](mono_audio, extraction_registry['key'][1])
            track_data['loudness'] = extraction_registry['loudness'][0](audio, extraction_registry['loudness'][1])
            track_data['effnet_indices'] = mean_effnet.tolist() # Mean vector for similarity
            
            # Classifiers use full embedding frames
            track_data['music_styles'] = extraction_registry['music_styles'][0](full_effnet, extraction_registry['music_styles'][1])
            track_data['voice_presence'] = extraction_registry['voice_presence'][0](full_effnet, extraction_registry['voice_presence'][1])
            track_data['danceability'] = extraction_registry['danceability'][0](full_effnet, extraction_registry['danceability'][1])
            # CLAP
            track_data['clap_indices'] = extraction_registry['TA_embeddings'][0](audio_48k, extraction_registry['TA_embeddings'][1])

            results[path] = track_data

            # Save incrementally every 10 files to prevent data loss on crash
            if len(results) % 10 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f)

        except Exception as e:
            print(f"\nError analyzing {path}: {e}. Skipping...")
            continue

    # Final Save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nAnalysis complete. Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audio_analysis.py <folder_path>")
        sys.exit(1)
    parse_folder(sys.argv[1])