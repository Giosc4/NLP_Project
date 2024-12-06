import nemo.collections.asr as nemo_asr
from difflib import SequenceMatcher
from nemo.collections.asr.metrics.wer import word_error_rate
import json
import torch
import os


class ASRInference:
    def __init__(self, model_path, val_manifest, use_gpu=True):
        # Load the ASR classification model
        self.model = nemo_asr.models.EncDecClassificationModel.restore_from(model_path)
        self.audio_paths = self.load_audio_paths(val_manifest)
        # Get labels from the model configuration
        self.labels = self.model.cfg.labels
        # Set the model to evaluation mode
        self.model.eval()
        if use_gpu and torch.cuda.is_available():
            self.model.cuda()

    @staticmethod
    def load_audio_paths(manifest_path):
        # Reads audio paths from the JSON manifest file
        with open(manifest_path, 'r') as f:
            return [json.loads(line)["audio_filepath"] for line in f]

    def transcribe_audio(self):
        # Esegue la trascrizione per i file audio specificati
        transcriptions = []
        with torch.no_grad():
            for audio_path in self.audio_paths:
                transcription = self.model.transcribe([audio_path])
                
                # Converti l'indice in una stringa di etichetta
                if isinstance(transcription[0], torch.Tensor):
                    index = transcription[0].item()  # Estrai l'indice dal tensore
                    predicted_label = self.labels[index] if 0 <= index < len(self.labels) else "indice_non_valido"
                else:
                    predicted_label = transcription[0]
                
                print(f"Raw model output for {audio_path}: {transcription}, Predicted Label: {predicted_label}")
                transcriptions.append(predicted_label)
        return transcriptions

    def extract_label_from_path(self, path):
        # Extracts the correct label from the file name
        base = os.path.basename(path)
        try:
            label = base.split('_')[1].split('.')[0]
            return label.lower()
        except IndexError:
            print(f"Error extracting label from file: {base}")
            return None
    
    def calculate_cer(self, hypotheses, references):
        """
        Calcola il CER (Character Error Rate).
        """
        total_chars = sum(len(ref) for ref in references)
        total_errors = sum(
            len(ref) + len(hyp) - 2 * sum(block.size for block in SequenceMatcher(None, ref, hyp).get_matching_blocks())
            for ref, hyp in zip(references, hypotheses)
        )
        return total_errors / total_chars if total_chars > 0 else 0

    def display_results(self):
        """
        Mostra i risultati di trascrizione e calcola WER e CER.
        """
        print("========================")
        print("Inizio Trascrizione e Valutazione")
        print("========================")
        
        # Trascrivi gli audio
        transcriptions = self.transcribe_audio()
        correct = []
        incorrect = []

        for idx, (path, transcription) in enumerate(zip(self.audio_paths, transcriptions), start=1):
            # Estrai il ground truth dall'etichetta
            ground_truth = self.extract_label_from_path(path)
            if ground_truth is None:
                continue

            # Determina il label predetto
            predicted_label = transcription.lower() if isinstance(transcription, str) else "formato_non_riconosciuto"

            # Confronta label predetto e ground truth
            if predicted_label == ground_truth:
                correct.append((path, predicted_label))
            else:
                incorrect.append((path, ground_truth, predicted_label))

            # Mostra i risultati per file
            print(f"File {idx}: {path}")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Prediction:   {predicted_label}")
            print("---------------------------")

        # Calcola statistiche di accuratezza
        total = len(correct) + len(incorrect)
        accuracy = (len(correct) / total) * 100 if total > 0 else 0

        # Mostra statistiche generali
        print("========================")
        print("Risultati Generali")
        print("========================")
        print(f"Total files analyzed: {total}")
        print(f"Correct predictions: {len(correct)}")
        print(f"Incorrect predictions: {len(incorrect)}")
        print(f"Accuracy percentage: {accuracy:.2f}%\n")

        # Carica WER e CER
        ground_truths = [self.extract_label_from_path(path) for path in self.audio_paths]
        wer = word_error_rate(hypotheses=transcriptions, references=ground_truths)
        cer = self.calculate_cer(transcriptions, ground_truths)

        # Mostra metriche di valutazione
        print("========================")
        print("Metriche di Valutazione")
        print("========================")
        print(f"Word Error Rate (WER): {wer:.2f}")
        print(f"Character Error Rate (CER): {cer:.2f}")


# Parameters for inference
model_path = "../asr_model.nemo"
val_manifest = "../val_manifest.json"

# Perform inference
inference = ASRInference(model_path, val_manifest)
inference.display_results()

