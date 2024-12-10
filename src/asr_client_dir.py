import os
import requests
from pydub import AudioSegment

# Configura il server Flask
SERVER_URL = "http://127.0.0.1:5001/predict"

# Directory contenente i file audio da processare
INPUT_DIR = "../renamed_audio_files"
OUTPUT_DIR = "../registratipt1"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Crea la directory di output se non esiste

def check_audio_file(file_path):
    """
    Controlla se il file audio è valido (esiste, non vuoto, formato corretto).
    """
    if not os.path.exists(file_path):
        print(f"Errore: il file {file_path} non esiste.")
        return False

    try:
        audio = AudioSegment.from_file(file_path)
        print(f"File {file_path}: {audio.frame_rate} Hz, {audio.channels} canali")
        if audio.frame_rate != 16000 or audio.channels != 1:
            print("Errore: il file non è in formato 16kHz mono")
            return False
    except Exception as e:
        print(f"Errore durante l'analisi del file {file_path}: {e}")
        return False

    return True

def send_audio_file(file_path):
    """
    Invia un file audio al server e restituisce la predizione del comando.
    """
    try:
        if not check_audio_file(file_path):
            print(f"File non valido: {file_path}")
            return "Invalid"

        with open(file_path, 'rb') as f:
            files = {'file': ('audio.wav', f, 'audio/wav')}
            response = requests.post(SERVER_URL, files=files)
            
            print(f"Stato risposta: {response.status_code}")
            print(f"Risposta: {response.text}")
            
            if response.status_code == 200:
                return response.json().get('command', 'Unknown')
            else:
                return "Error"
    except Exception as e:
        print(f"Errore durante l'invio del file {file_path}: {e}")
        return "Error"


def process_audio_files(input_dir, output_dir):
    """
    Processa tutti i file audio nella directory di input.
    Rinominare i file con il comando predetto e un numero incrementale.
    """
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    counter = 1

    for audio_file in audio_files:
        file_path = os.path.join(input_dir, audio_file)
        print(f"Processando file: {file_path}")

        # Invia il file al server e ottieni la predizione
        command = send_audio_file(file_path)
        print(f"Predizione per {audio_file}: {command}")

        # Genera il nuovo nome del file
        new_file_name = f"{command}_{counter}.wav"
        new_file_path = os.path.join(output_dir, new_file_name)
        counter += 1

        # Copia o rinomina il file nella directory di output
        if command != "Invalid":
            os.rename(file_path, new_file_path)
            print(f"File rinominato e spostato in: {new_file_path}")
        else:
            print(f"File non valido: {file_path}")

if __name__ == "__main__":
    process_audio_files(INPUT_DIR, OUTPUT_DIR)
