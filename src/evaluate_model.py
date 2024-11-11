import nemo.collections.asr as nemo_asr
import json

class ASRInference:
    def __init__(self, model_path, val_manifest, labels):
        """
        Classe per eseguire inferenze su file audio con un modello ASR addestrato.

        :param model_path: Percorso al modello salvato.
        :param val_manifest: Percorso al file manifesto di valutazione contenente i percorsi degli audio.
        :param labels: Lista delle etichette previste dal modello.
        """
        self.model = nemo_asr.models.EncDecClassificationModel.restore_from(model_path)
        self.audio_paths = self.load_audio_paths(val_manifest)
        self.labels = labels

    @staticmethod
    def load_audio_paths(manifest_path):
        """
        Carica i percorsi dei file audio dal file manifesto.

        :param manifest_path: Percorso al manifesto.
        :return: Lista dei percorsi dei file audio.
        """
        with open(manifest_path, 'r') as f:
            return [json.loads(line)["audio_filepath"] for line in f]

    def transcribe_audio(self):
        """
        Esegue la trascrizione dei file audio e restituisce le etichette predette.

        :return: Lista di dizionari con percorso del file audio, indice e etichetta prevista.
        """
        transcriptions = self.model.transcribe(paths2audio_files=self.audio_paths)
        results = []
        
        for path, transcription in zip(self.audio_paths, transcriptions):
            predicted_index = transcription.item()  # Indice della previsione
            predicted_label = self.labels[predicted_index]  # Etichetta associata
            
            results.append({
                "audio_filepath": path,
                "predicted_index": predicted_index,
                "predicted_label": predicted_label
            })
        return results

    def display_results(self):
        """
        Stampa i risultati delle trascrizioni con etichetta interpretata per ogni file audio.
        """
        results = self.transcribe_audio()
        
        for i, result in enumerate(results):
            print("------------------------")
            print(f"File Audio {i+1}: {result['audio_filepath']}")
            print("Trascrizione interpretata:", result["predicted_label"])


# Parametri per l'inferenza
model_path = "/home/giolinux/NLP_Project/asr_model_simple.nemo"
val_manifest = "/home/giolinux/NLP_Project/val_manifest.json"
labels = [
    "avanti", "cammina", "continua", "corri", "destra", "esci",
    "fermo", "giu", "indietro", "pausa", "salta", "sinistra", "su", "vola"
]

# Esegui inferenza
inference = ASRInference(model_path, val_manifest, labels)
inference.display_results()
