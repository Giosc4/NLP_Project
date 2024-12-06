from flask import Flask, request, jsonify
from nemo.collections.asr.models import ASRModel
import os
import uuid

app = Flask(__name__)

# Carica il modello .nemo
print("Caricamento del modello...")
model = ASRModel.restore_from("../asr_model.nemo")
print("Modello caricato correttamente.")

labels = [
    "avanti", "indietro", "sinistra", "destra",
    "cammina", "corri", "fermo", "salta",
    "vola", "su", "giu", "pausa",
    "continua", "esci"
]


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

        # Rimuovi il file temporaneo
        os.remove(temp_audio_path)
        print("File audio temporaneo rimosso.")

        return jsonify({'command': command})

    except Exception as e:
        print(f"Errore durante l'elaborazione: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
