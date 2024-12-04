import nemo.collections.asr as nemo_asr
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
        # Performs transcription for the specified audio files
        transcriptions = []
        with torch.no_grad():
            for audio_path in self.audio_paths:
                transcription = self.model.transcribe([audio_path])
                # Print raw model output for debugging
                print(f"Raw model output for {audio_path}: {transcription}")
                transcriptions.append(transcription[0])
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

    def display_results(self):
        # Prints the transcription results comparing with the correct labels
        transcriptions = self.transcribe_audio()
        correct = []
        incorrect = []

        for path, transcription in zip(self.audio_paths, transcriptions):
            ground_truth = self.extract_label_from_path(path)
            if ground_truth is None:
                continue

            # Extract the index from the prediction and map to the label
            if isinstance(transcription, torch.Tensor):
                index = transcription.argmax().item()
                if 0 <= index < len(self.labels):
                    predicted_label = self.labels[index].lower()
                else:
                    predicted_label = "indice_non_valido"
            elif isinstance(transcription, list) or isinstance(transcription, np.ndarray):
                index = int(transcription[0])
                if 0 <= index < len(self.labels):
                    predicted_label = self.labels[index].lower()
                else:
                    predicted_label = "indice_non_valido"
            elif isinstance(transcription, str):
                predicted_label = transcription.strip().lower()
            else:
                predicted_label = "formato_non_riconosciuto"

            # Compare the prediction with the correct label
            if predicted_label == ground_truth:
                correct.append((path, predicted_label))
            else:
                incorrect.append((path, ground_truth, predicted_label))

            # Print for debugging
            print(f"File: {path}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Model Prediction: {predicted_label}")
            print("---------------------------")

        # Calculate the accuracy percentage
        total = len(correct) + len(incorrect)
        accuracy = (len(correct) / total) * 100 if total > 0 else 0

        # Print the results
        print("========================")
        print("Transcription Results")
        print("========================")
        print(f"Total files analyzed: {total}")
        print(f"Correct predictions: {len(correct)}")
        print(f"Incorrect predictions: {len(incorrect)}")
        print(f"Accuracy percentage: {accuracy:.2f}%\n")

        if correct:
            print("== Correct Predictions ==")
            for i, (path, label) in enumerate(correct, 1):
                print(f"{i}. {path} --> {label}")
            print("\n")

        if incorrect:
            print("== Incorrect Predictions ==")
            for i, (path, ground, pred) in enumerate(incorrect, 1):
                print(f"{i}. {path}")
                print(f"   Correct Label: {ground}")
                print(f"   Model Prediction: {pred}\n")

# Parameters for inference
model_path = "/home/giova/NLP/NLP_Project-main/asr_model.nemo"
val_manifest = "/home/giova/NLP/NLP_Project-main/val_manifest.json"

# Perform inference
inference = ASRInference(model_path, val_manifest)
inference.display_results()
