from flask import Flask, request, jsonify
from nemo.collections.asr.models import ASRModel
import os
import uuid
import shutil

app = Flask(__name__)

# Carica il modello .nemo
print("Caricamento del modello...")
model = ASRModel.restore_from("../asr_model2.nemo")
print("Modello caricato correttamente.")

labels = [
    "avanti", "indietro", "sinistra", "destra",
    "cammina", "corri", "fermo", "salta",
    "vola", "su", "giu", "pausa",
    "continua", "esci"
]

# Directory di destinazione per salvare gli audio
SAVED_AUDIO_DIR = "./saved_audio"
os.makedirs(SAVED_AUDIO_DIR, exist_ok=True)  # Crea la cartella se non esiste

def generate_unique_filename(base_name, extension, directory):
    """Genera un nome unico per il file nella directory specificata."""
    counter = 1
    file_name = f"{base_name}.{extension}"
    file_path = os.path.join(directory, file_name)
    
    while os.path.exists(file_path):
        file_name = f"{base_name}_{counter}.{extension}"
        file_path = os.path.join(directory, file_name)
        counter += 1

    return file_path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Richiesta ricevuta.")
        
        # Ottieni il file audio dalla richiesta
        if 'file' not in request.files:
            print("File audio mancante nella richiesta.")
            return jsonify({'error': 'File audio mancante'}), 400
        
        audio_file = request.files['file']
        temp_audio_path = f"/tmp/{uuid.uuid4()}.wav"
        audio_file.save(temp_audio_path)
        print(f"File audio salvato temporaneamente in: {temp_audio_path}")

        # Esegui la trascrizione
        print("Inizio trascrizione...")
        predictions = model.transcribe([temp_audio_path])  # Restituisce una lista di tensori
        print(f"Predizioni ricevute: {predictions}")

        # Converte il tensore in indice e poi in comando
        if predictions and len(predictions) > 0:
            index = predictions[0].item()  # Estrai il valore numerico dal tensore
            command = labels[index] if index < len(labels) else "Unknown"
        else:
            command = "Unknown"

        # Genera un nuovo nome per l'audio da salvare definitivamente
        base_name = command
        saved_audio_path = generate_unique_filename(base_name, "wav", SAVED_AUDIO_DIR)
        shutil.move(temp_audio_path, saved_audio_path)
        print(f"File audio spostato in: {saved_audio_path}")

        return jsonify({'command': command})

    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
