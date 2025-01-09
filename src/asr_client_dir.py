import os

# Lista di parole valide
valid_words = [
    "avanti", "indietro", "sinistra", "destra",
    "cammina", "corri", "fermo", "salta",
    "vola", "su", "giu", "pausa",
    "continua", "esci"
]

# Specifica la directory contenente i file
directory = "/home/giova/NLP/NLP_Project-main/audio"  # Modifica il percorso della directory


def check_file_names(dir_path, valid_words):
    """
    Controlla se le parole nei nomi dei file appartengono alla lista di parole valide.
    """
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} non trovata!")
        return

    for root, dirs, files in os.walk(dir_path):
            for filename in files:
                # Considera solo file con estensione .wav
                if filename.endswith(".wav"):
                    # Estrai la parola dal nome del file (assumendo formato numero_parola.wav)
                    parts = filename.split("_")
                    if len(parts) >= 2:  # Assicurati che il formato sia valido
                        word = parts[1].split(".")[0]  # Ottieni la parola dopo l'underscore
                        if word in valid_words:
                            print(f".")
                        else:
                            print(f"'{filename}' contiene una parola NON valida: {word} (Percorso: {root})")
                    else:
                        print(f"Formato del file non valido: {filename} (Percorso: {root})")
                else:
                    print(f"File ignorato (non .wav): {filename} (Percorso: {root})")

# Esegui la funzione sulla directory specificata
check_file_names(directory, valid_words)