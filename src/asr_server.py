from flask import Flask, request, jsonify
from nemo.collections.asr.models import EncDecCTCModel
import os

# Inizializza il server Flask
app = Flask(__name__)

# Carica il modello .nemo
model = EncDecCTCModel.restore_from("../asr_model.nemo")

# Lista delle etichette corrispondenti agli indici
labels = [
    "avanti", "indietro", "sinistra", "destra",
    "cammina", "corri", "fermo", "salta",
    "vola", "su", "giu", "pausa",
    "continua", "esci"
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ottieni il file audio dalla richiesta
        audio_file = request.files['file']
        
        # Salva il file in una directory temporanea
        temp_audio_path = "/tmp/temp_audio.wav"
        audio_file.save(temp_audio_path)
        
        # Esegui la trascrizione
        predictions = model.transcribe([temp_audio_path])  # Restituisce una lista
        
        # Ottieni il tensore dalla predizione e convertilo in un indice numerico
        if isinstance(predictions, list) and len(predictions) > 0:
            tensor_prediction = predictions[0]
            
            # Estrai il valore dal tensore e ottieni la label
            if hasattr(tensor_prediction, 'item'):  # Controllo se è un tensore PyTorch
                index = int(tensor_prediction.item())
            else:
                index = int(tensor_prediction)  # Se è già un valore numerico
            
            # Mappa l'indice alla label corrispondente
            command = labels[index] if index < len(labels) else "Unknown"
        else:
            command = "Unknown"
        
        # Rimuovi il file temporaneo
        os.remove(temp_audio_path)
        
        # Restituisci il comando trascritto
        return jsonify({'command': command})
    except Exception as e:
        # Gestione errori
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
