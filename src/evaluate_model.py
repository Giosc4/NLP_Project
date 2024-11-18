import nemo.collections.asr as nemo_asr
import json
import torch
import os

class ASRInference:
    def __init__(self, model_path, val_manifest, labels, use_gpu=True):
        # Carica il modello ASR
        self.model = nemo_asr.models.EncDecCTCModel.restore_from(model_path)
        self.audio_paths = self.load_audio_paths(val_manifest)
        self.labels = labels
        # Imposta il modello in modalità di valutazione
        self.model.eval()
        if use_gpu and torch.cuda.is_available():
            self.model.cuda()

    @staticmethod
    def load_audio_paths(manifest_path):
        # Legge i percorsi audio dal file manifest JSON
        with open(manifest_path, 'r') as f:
            return [json.loads(line)["audio_filepath"] for line in f]

    def transcribe_audio(self):
        # Esegue la trascrizione per i file audio specificati
        with torch.no_grad():  # Disabilita il calcolo dei gradienti per ottimizzare la memoria
            # Usa il metodo transcribe() corretto per la tua versione di NeMo
            transcriptions = self.model.transcribe(self.audio_paths)
        return transcriptions

    def extract_label_from_path(self, path):
        # Estrae l'etichetta corretta dal nome del file
        base = os.path.basename(path)
        # Supponendo che il formato sia 'sara2_continua.wav'
        try:
            label = base.split('_')[1].split('.wav')[0]
            return label.lower()
        except IndexError:
            return None

    def display_results(self):
        # Stampa i risultati della trascrizione confrontando con le etichette corrette
        transcriptions = self.transcribe_audio()
        correct = []
        incorrect = []

        for path, transcription in zip(self.audio_paths, transcriptions):
            ground_truth = self.extract_label_from_path(path)
            if ground_truth is None:
                print(f"Errore nell'estrazione dell'etichetta per il file: {path}")
                continue

            # Mappa la predizione
            if isinstance(transcription, torch.Tensor):
                label_index = transcription.item()
                if 0 <= label_index < len(self.labels):
                    predicted_label = self.labels[label_index].lower()
                else:
                    predicted_label = "Indice non valido"
            elif isinstance(transcription, str):
                predicted_label = transcription.lower()
            else:
                predicted_label = "Formato non riconosciuto"

            # Confronta la predizione con l'etichetta corretta
            if predicted_label == ground_truth:
                correct.append((path, predicted_label))
            else:
                incorrect.append((path, ground_truth, predicted_label))

        # Stampa i risultati
        print("========================")
        print("Risultati della Trascrizione")
        print("========================")
        print(f"Totale file analizzati: {len(self.audio_paths)}")
        print(f"Predizioni corrette: {len(correct)}")
        print(f"Predizioni sbagliate: {len(incorrect)}\n")
        print(f"Percentuale di accuratezza: {len(correct) / len(self.audio_paths) * 100:.2f}%\n")


        if correct:
            print("== Predizioni Corrette ==")
            for i, (path, label) in enumerate(correct, 1):
                print(f"{i}. {path} --> {label}")
            print("\n")

        if incorrect:
            print("== Predizioni Sbagliate ==")
            for i, (path, ground, pred) in enumerate(incorrect, 1):
                print(f"{i}. {path}")
                print(f"   Etichetta Corretta: {ground}")
                print(f"   Predizione Modello: {pred}\n")

# Parametri per l'inferenza
model_path = "/home/giolinux/NLP/NLP_Project/asr_model.nemo"
val_manifest = "/home/giolinux/NLP/NLP_Project/val_manifest.json"
labels = [
    "avanti", "cammina", "continua", "corri", "destra", "esci",
    "fermo", "giu", "indietro", "pausa", "salta", "sinistra", "su", "vola"
]

# Esegui inferenza
inference = ASRInference(model_path, val_manifest, labels)
inference.display_results()
