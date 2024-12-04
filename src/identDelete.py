import os

def delete_identifier_files(directory):
    """
    Elimina tutti i file che terminano con '.Identifier' in una directory specificata.

    :param directory: Percorso della directory in cui cercare i file.
    """
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.Identifier'):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Eliminato: {file_path}")
        print("Operazione completata.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

# Specifica la directory da pulire
directory_to_clean = r"/home/giova/NLP/NLP_Project-main"  # Usa forward slash o una stringa raw
delete_identifier_files(directory_to_clean)
