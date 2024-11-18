import os
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
import subprocess

def add_background_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data

def time_shift(y, shift_max=0.2):
    shift = np.random.randint(int(len(y) * shift_max))
    if np.random.randint(2):
        shift = -shift
    augmented_data = np.roll(y, shift)
    return augmented_data

def change_pitch_and_speed(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def apply_lowpass_filter(y, sr, cutoff=3000):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(1, norm_cutoff, btype='low', analog=False)
    y_filtered = lfilter(b, a, y)
    return y_filtered

def apply_highpass_filter(y, sr, cutoff=500):
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    b, a = butter(1, norm_cutoff, btype='high', analog=False)
    y_filtered = lfilter(b, a, y)
    return y_filtered

def augment_data(input_dir, output_dir, sample_rate=16000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                y, sr = librosa.load(file_path, sr=sample_rate)
                
                # Crea la stessa struttura di directory nell'output
                rel_dir = os.path.relpath(subdir, input_dir)
                output_subdir = os.path.join(output_dir, rel_dir)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                
                # Salva il file originale nella directory di output
                output_file = os.path.join(output_subdir, file)
                sf.write(output_file, y, sr)
                
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
                                   

if __name__ == "__main__":

    # Esegui il primo script di data augmentation
    script1 = '/home/giolinux/NLP/NLP_Project/src/manifest.py'
    result1 = subprocess.run(['python3', script1], capture_output=True, text=True)
    print(f"Uscita di {script1}:\n{result1.stdout}")
    if result1.stderr:
        print(f"Errori di {script1}:\n{result1.stderr}")
    input_directory = '/home/giolinux/NLP/NLP_Project/audio'          # Sostituisci con il tuo percorso di input
    output_directory = '/home/giolinux/NLP/NLP_Project/augmented_audio'  # Sostituisci con il tuo percorso di output
    augment_data(input_directory, output_directory)
