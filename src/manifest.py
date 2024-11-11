import os
import json
import soundfile as sf
from sklearn.model_selection import train_test_split

# Directory principale contenente tutte le sottodirectory con i file audio
main_audio_dir = "/home/giolinux/NLP_Project/audio/"
manifest_path = "/home/giolinux/NLP_Project/data_manifest.json"
train_manifest = "/home/giolinux/NLP_Project/train_manifest.json"
val_manifest = "/home/giolinux/NLP_Project/val_manifest.json"

# Lista che conterrà i dati per ogni file
data_entries = []

# Esplora tutte le directory e sottodirectory
for root, dirs, files in os.walk(main_audio_dir):
    for filename in files:
        # Verifica che il file sia un file .wav
        if filename.endswith(".wav"):
            filepath = os.path.join(root, filename)
            try:
                # Calcola la durata dell'audio
                with sf.SoundFile(filepath) as audio_file:
                    duration = len(audio_file) / audio_file.samplerate

                # Estrai la parola detta (dopo l'underscore) e rimuovi l'estensione
                word_spoken = filename.split('_')[-1].replace('.wav', '')

                # Crea un dizionario per il file corrente
                entry = {
                    "audio_filepath": filepath,
                    "duration": duration,
                    "label": word_spoken
                }
                data_entries.append(entry)

            except RuntimeError:
                print(f"File {filepath} non accessibile. Ignorato.")

# Salva il manifest completo con tutti i dati
with open(manifest_path, 'w') as manifest_file:
    for entry in data_entries:
        json.dump(entry, manifest_file)
        manifest_file.write('\n')

print(f"Manifest file completo creato con successo in {manifest_path}")

# Divide i dati in train e validation set con divisione randomica 90-10
train_entries, val_entries = train_test_split(data_entries, test_size=0.1, random_state=None, shuffle=True)

# Salva i manifest separati per train e validation
with open(train_manifest, 'w') as train_file:
    for entry in train_entries:
        json.dump(entry, train_file)
        train_file.write('\n')

with open(val_manifest, 'w') as val_file:
    for entry in val_entries:
        json.dump(entry, val_file)
        val_file.write('\n')    

print("I dati sono stati divisi nei set di addestramento e validazione.")
