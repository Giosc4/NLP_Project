# ASR per il Controllo di un Avatar 3D con Comandi Vocali

## Introduzione
Questo progetto implementa un sistema **Automatic Speech Recognition (ASR)** progettato per riconoscere **comandi vocali in lingua italiana** e utilizzarli per controllare un avatar all'interno di un ambiente 3D creato con Unity. Il progetto combina tecniche avanzate di ASR con un'integrazione fluida in Unity per offrire un'interfaccia vocale intuitiva e accessibile.

## Scopo del Progetto
Lo scopo del progetto è sviluppare un sistema ASR che riconosca **14 comandi vocali specifici** e traduca questi comandi in azioni eseguite da un avatar in un ambiente virtuale.

## Struttura del Progetto
Il progetto è organizzato come segue:

- **Dataset**: Contiene file audio personalizzati per il training del modello ASR.
- **Manifest JSON**: File necessari per addestrare il modello NeMo.
- **Modello**: Modello pre-addestrato (`asr_model.nemo`) e ottimizzato (`optimized_asr_model.nemo`).
- **Unity Client**: Script C# che registra l'audio, lo invia al server ASR e interpreta la risposta.
- **Server ASR**: Servizio Flask che ospita il modello ASR per l'elaborazione degli audio.
- **Codice**: Script Python per il preprocessing, l'addestramento e l'integrazione con NeMo.

## Strumenti e Tecnologie Utilizzati
- **Python**: Linguaggio di programmazione principale.
- **NVIDIA NeMo**: Libreria per l'addestramento del modello ASR.
- **PyTorch**: Framework di machine learning utilizzato da NeMo.
- **Flask**: Framework leggero per esporre il modello ASR come servizio REST.
- **Unity**: Ambiente grafico 3D per il controllo e il testing dell'avatar.
- **Audacity**: Software per il preprocessing dei file audio.
- **Optuna**: Per l'ottimizzazione automatizzata degli iperparametri.
- **Librosa**: Libreria per il processing audio.
- **Soundfile**: Gestione avanzata dei file audio.
- **Scipy**: Filtraggio dei segnali audio.

## Dataset
A causa della mancanza di risorse italiane esistenti, è stato creato un **dataset personalizzato**:

- **Formato audio**: WAV a 16kHz, mono.
- **Comandi vocali**: 14 parole (es: "avanti", "indietro", "sinistra", "pausa").
- **Data Augmentation**: Rumore di fondo, variazioni temporali, pitch shift, filtri passa-basso/alto.
- **Manifest JSON**: Percorsi, etichette e durata dei file audio necessari per l'addestramento.

## Comandi Supportati
Il sistema supporta i seguenti comandi vocali:

- **"avanti"**: Sposta l'avatar in avanti.
- **"indietro"**: Sposta l'avatar all'indietro.
- **"sinistra"**: Ruota l'avatar a sinistra.
- **"destra"**: Ruota l'avatar a destra.
- **"salta"**: Fa saltare l'avatar.
- **"vola"**: Attiva la modalità di volo.
- **"su"**: Sposta l'avatar in alto (in volo).
- **"giù"**: Sposta l'avatar in basso (in volo).
- **"fermo"**: Ferma tutti i movimenti dell'avatar.
- **"corri"**: Attiva la modalità di corsa.
- **"cammina"**: Attiva la camminata normale.
- **"pausa"**: Mette il gioco in pausa.
- **"continua"**: Riprende il gioco dopo la pausa.
- **"esci"**: Chiude l'applicazione Unity.

## Installazione e Configurazione
### Prerequisiti
- **Python 3.8+**
- **PyTorch e NVIDIA NeMo**
- **Unity** (versione consigliata: Unity 2021+)
- **Flask**
- Microfono per la registrazione dei comandi vocali

### Installazione
1. Clona il repository:
   ```bash
   git clone https://github.com/Giosc4/NLP_Project.git
   cd NLP_Project
   ```
2. Installa le dipendenze Python:
   ```bash
   pip install -r requirements.txt
   ```
   Il file `requirements.txt` deve includere:
   ```
   flask
   torch
   pytorch-lightning
   nvidia-pyindex
   nemo-toolkit[asr]
   librosa
   optuna
   soundfile
   scipy
   ```
3. Avvia il server ASR:
   ```bash
   python aser_server.py
   ```
   Il server verrà avviato all'indirizzo `http://localhost:5001`.

### Configurazione in Unity
1. Apri il progetto Unity contenuto nella cartella `UnityNLP_Script` all'indirizzo "https://github.com/Giosc4/UnityNLP_Script".
2. Aggiungi lo script `VoiceCommandHandler.cs` al tuo GameObject principale.
3. Assicurati che l'URL API sia configurato correttamente:
   ```csharp
   public string apiURL = "http://localhost:5001/predict";
   ```
4. Premi **Play** e utilizza il tasto `V` per registrare i comandi vocali.

## Testing del Sistema
Durante la fase di testing, tutti i comandi vocali inviati vengono salvati in una cartella dedicata e successivamente integrati nel dataset per migliorare continuamente il modello ASR.

## Risultati
- **Accuratezza del Modello**: 87% sui comandi vocali validati.
- **Error Rate**: WER del 13\%, CER del 9\%.
- **Performance**: Controllo fluido e reattivo dell'avatar.

## Possibili Sviluppi Futuri
- Ampliamento del dataset con ulteriori registrazioni vocali.
- Sperimentazione con nuove architetture di modelli ASR.
- Integrazione di ulteriori comandi e ottimizzazione delle animazioni Unity.
- Introduzione di modelli basati su **Transformer** per migliorare ulteriormente la precisione.

## Conclusioni
Questo progetto rappresenta un esempio concreto di come l'intelligenza artificiale possa essere integrata in ambienti virtuali per creare interfacce naturali e intuitive. L'utilizzo di NVIDIA NeMo e Unity ha consentito di sviluppare un sistema robusto, efficiente e facilmente estensibile.

## Crediti
- **NVIDIA NeMo**: Framework utilizzato per l'ASR.
- **PyTorch**: Backend per il training del modello.
- **Unity**: Motore grafico per l'ambiente virtuale.
- **Optuna**: Per l'ottimizzazione degli iperparametri.
- **ChatGPT** e **Windsurf AI**: Supporto nella progettazione e nello sviluppo del progetto.

## Riferimenti
- **NVIDIA NeMo**: [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **Unity**: [https://docs.unity3d.com/](https://docs.unity3d.com/)
- **Optuna**: [https://optuna.org/](https://optuna.org/)

---

Se hai domande o suggerimenti, sentiti libero di aprire un'**issue** o contattarmi!
