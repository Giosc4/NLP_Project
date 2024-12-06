import os
import json
import librosa
import random

def create_manifest(data_dir, manifest_path):
    with open(manifest_path, 'w') as manifest_file:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    # Estrai l'etichetta dal nome del file dopo il primo underscore
                    if "_" in file:
                        label = file.split("_")[1].split(".")[0]
                    else:
                        label = os.path.basename(os.path.dirname(file_path))
                    
                    # Calcola la durata del file audio
                    duration = librosa.get_duration(path=file_path)
                    entry = {
                        'audio_filepath': file_path,
                        'duration': duration,
                        'label': label
                    }
                    manifest_file.write(json.dumps(entry) + '\n')

def split_manifest(manifest_path, train_manifest_path, val_manifest_path, train_split=0.8):
    with open(manifest_path, 'r') as f:
        data = f.readlines()

    # Mischia casualmente i dati
    random.shuffle(data)

    # Calcola la divisione in 80% training e 20% validation
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    # Salva i dati di training
    with open(train_manifest_path, 'w') as f_train:
        f_train.writelines(train_data)

    # Salva i dati di validazione
    with open(val_manifest_path, 'w') as f_val:
        f_val.writelines(val_data)

    print(f"Divisione completata: {len(train_data)} campioni per il training, {len(val_data)} campioni per la validazione.")

# Directory dei file audio originali
data_dir = '../audio'
# Percorsi per i manifest
data_manifest_path = '../data_manifest.json'
train_manifest_path = '../train_manifest.json'
val_manifest_path = '../val_manifest.json'

# Crea il manifest principale con i dati originali
create_manifest(data_dir, data_manifest_path)
# Crea manifest di training e validation
split_manifest(data_manifest_path, train_manifest_path, val_manifest_path)
