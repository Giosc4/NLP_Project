import os
import json
import librosa

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
                        # Se non ci sono underscore, utilizza il nome della directory superiore come etichetta
                        label = os.path.basename(os.path.dirname(file_path))
                    
                    # Calcola la durata del file audio
                    duration = librosa.get_duration(path=file_path)
                    entry = {
                        'audio_filepath': file_path,
                        'duration': duration,
                        'label': label
                    }
                    manifest_file.write(json.dumps(entry) + '\n')

# Esempio di utilizzo:
data_dir = '/home/giolinux/NLP_Project/augmented_audio'  # Directory con i dati aumentati
manifest_path = '/home/giolinux/NLP_Project/train_manifest_augmented.json'
create_manifest(data_dir, manifest_path)
