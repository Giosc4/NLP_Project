from flask import Flask, request, jsonify
from nemo.collections.asr.models import ASRModel
import os
import uuid
import logging

app = Flask(__name__)

# Configura il logging
logging.basicConfig(level=logging.DEBUG)

# Carica il modello .nemo
logging.info("Caricamento del modello...")
model = ASRModel.restore_from("../asr_model.nemo")
logging.info("Modello caricato correttamente.")

labels = [
    "avanti", "indietro", "sinistra", "destra",
    "cammina", "corri", "fermo", "salta",
    "vola", "su", "giu", "pausa",
    "continua", "esci"
]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Richiesta ricevuta.")
        
        # Ottieni il file audio dalla richiesta
        if 'file' not in request.files:
            logging.error("File audio mancante nella richiesta.")
            return jsonify({'error': 'File audio mancante'}), 400
        
        audio_file = request.files['file']
        temp_audio_path = f"/tmp/{uuid.uuid4()}.wav"
        audio_file.save(temp_audio_path)
        logging.info(f"File audio salvato temporaneamente in: {temp_audio_path}")

        # Esegui la trascrizione
        logging.info("Inizio trascrizione...")
        predictions = model.transcribe([temp_audio_path])  # Restituisce una lista di tensori
        logging.info(f"Predizioni ricevute: {predictions}")

        # Converte il tensore in indice e poi in comando
        if predictions and len(predictions) > 0:
            index = predictions[0].item()  # Estrai il valore numerico dal tensore
            command = labels[index] if index < len(labels) else "Unknown"
        else:
            command = "Unknown"

        # Rimuovi il file temporaneo
        os.remove(temp_audio_path)
        logging.info("File audio temporaneo rimosso.")

        return jsonify({'command': command})

    except Exception as e:
        logging.error(f"Errore durante l'elaborazione: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
