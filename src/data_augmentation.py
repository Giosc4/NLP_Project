import os
import subprocess
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import json

def add_background_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return normalize_audio(augmented_data)

def time_shift(y, shift_max=0.2):
    shift = np.random.randint(int(len(y) * shift_max))
    if np.random.randint(2):
        shift = -shift
    augmented_data = np.roll(y, shift)
    return normalize_audio(augmented_data)

def change_pitch_and_speed(y, sr, n_steps=2):
    return normalize_audio(librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps))

def apply_lowpass_filter(y, sr, cutoff=3000):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(1, norm_cutoff, btype='low', analog=False)
    y_filtered = lfilter(b, a, y)
    return normalize_audio(y_filtered)

def apply_highpass_filter(y, sr, cutoff=500):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(1, norm_cutoff, btype='high', analog=False)
    y_filtered = lfilter(b, a, y)
    return normalize_audio(y_filtered)

def normalize_audio(y):
    return y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y

def update_manifest(manifest_path, file_path, duration, label):
    """Aggiunge un file al manifest."""
    entry = {
        "audio_filepath": file_path,
        "duration": duration,
        "label": label
    }
    with open(manifest_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

def extract_label_from_path(file_path):
    """Estrae il label dal percorso del file."""
    filename = os.path.basename(file_path)  # Estrai il nome del file (es. 'gio1_cammina.wav')
    label = filename.split('_')[1].split('.')[0]  # Estrai la parte tra l'underscore e l'estensione
    return label

def augment_data(input_dir, output_dir, manifest_path, sample_rate=16000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                try:
                    y, sr = librosa.load(file_path, sr=sample_rate)
                    duration = librosa.get_duration(y=y, sr=sr)
                except Exception as e:
                    print(f"Errore nel caricamento del file {file_path}: {e}")
                    continue
                
                # Estrai l'etichetta dal nome del file (ad esempio "label.wav")
                label = extract_label_from_path(file_path)


                # Crea la stessa struttura di directory nell'output
                rel_dir = os.path.relpath(subdir, input_dir)
                output_subdir = os.path.join(output_dir, rel_dir)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Salva il file originale nella directory di output
                output_file = os.path.join(output_subdir, file)
                sf.write(output_file, y, sr)
                update_manifest(manifest_path, output_file, duration, label)
                
                # Lista di tecniche di data augmentation
                augmentations = [
                    ('_noise', add_background_noise(y)),
                    ('_timeshift', time_shift(y)), 
                    ('_pitch', change_pitch_and_speed(y, sr)),
                    ('_lowpass', apply_lowpass_filter(y, sr)),
                    ('_highpass', apply_highpass_filter(y, sr)),
                ]
                
                # Applica e salva ogni data augmentation
                for aug_suffix, aug_data in augmentations:
                    aug_file = file.replace('.wav', f'{aug_suffix}.wav')
                    aug_file_path = os.path.join(output_subdir, aug_file)
                    sf.write(aug_file_path, aug_data, sr)
                    update_manifest(manifest_path, aug_file_path, duration, label)

if __name__ == "__main__":
    input_directory = '../audio'    
    output_directory = '../augmented_audio'
    manifest_path = '../train_manifest_augmented.json'

    # Esegui il primo script di data augmentation
    script1 = 'manifest.py'
    result1 = subprocess.run(['python3', script1], capture_output=True, text=True)
    print(f"Uscita di {script1}:\n{result1.stdout}")
    if result1.stderr:
        print(f"Errori di {script1}:\n{result1.stderr}")
    
    # Crea un nuovo manifest o svuota il vecchio
    open(manifest_path, 'w').close()
    
    print("[INFO] Inizio del processo di data augmentation...")
    augment_data(input_directory, output_directory, manifest_path)
    print("[INFO] Processo di data augmentation completato. Manifest aggiornato!")
