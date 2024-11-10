
# import nemo
# import nemo.collections.asr as nemo_asr
import os
import json
import soundfile as sf

# Directory corretta dei file audio
audio_dir = "/home/giolinux/NLP_Project/audio/greta1/"
manifest_path = "data_manifest.json"

# Lista che conterrà i dati per ogni file
data_entries = []

# Itera su ogni file nella directory
for filename in os.listdir(audio_dir):
    # Verifica che il file sia un file .wav
    if filename.endswith(".wav"):
        # Estrai la parola detta (dopo l'underscore) e rimuovi l'estensione
        word_spoken = filename.split('_')[-1].replace('.wav', '')
        
        # Percorso completo del file
        filepath = os.path.join(audio_dir, filename)
        
        # Calcola la durata dell'audio
        with sf.SoundFile(filepath) as audio_file:
            duration = len(audio_file) / audio_file.samplerate
        
        # Crea un dizionario per il file corrente
        entry = {
            "audio_filepath": filepath,
            "duration": duration,
            "label": word_spoken
        }
        
        # Aggiungi l'entry alla lista
        data_entries.append(entry)

# Scrivi il manifest in un file JSON
with open(manifest_path, 'w') as manifest_file:
    for entry in data_entries:
        json.dump(entry, manifest_file)
        manifest_file.write('\n')  # Aggiungi una nuova riga tra le entries

print(f"Manifest file creato con successo in {manifest_path}")
