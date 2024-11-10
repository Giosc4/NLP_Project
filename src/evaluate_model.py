import nemo.collections.asr as nemo_asr
import json

# Percorso al modello salvato e al manifesto di valutazione
model_path = "asr_model.nemo"
val_manifest = "val_manifest.json"

# Carica il modello addestrato
asr_model = nemo_asr.models.EncDecClassificationModel.restore_from(model_path)

# Carica i percorsi dei file audio dal manifesto
with open(val_manifest, 'r') as f:
    audio_paths = [json.loads(line)["audio_filepath"] for line in f]

# Lista delle etichette in base all'ordine del tuo training
labels = [
    "avanti", "cammina", "continua", "corri", "destra", "esci",
    "fermo", "giu", "indietro", "pausa", "salta", "sinistra", "su", "vola"
]

# Esegui la trascrizione su ciascun file audio
transcriptions = asr_model.transcribe(paths2audio_files=audio_paths)

# Visualizza le trascrizioni per ogni file audio con interpretazione delle etichette
for i, transcription in enumerate(transcriptions):
    predicted_index = transcription.item()  # Ottieni l'indice della previsione come intero
    predicted_label = labels[predicted_index]  # Associa l'indice all'etichetta corretta
    
    print(f"File Audio {i+1}: {audio_paths[i]}")
    print("Indice Previsione:", predicted_index)
    print("Trascrizione interpretata:", predicted_label)
