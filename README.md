# **ASR per il Controllo di un Avatar 3D con Comandi Vocali**

## **Introduzione**
Questo progetto implementa un sistema di **Automatic Speech Recognition (ASR)** per riconoscere **comandi vocali in lingua italiana** e tradurli in azioni per controllare un avatar 3D in Unity. L'obiettivo è fornire un'interfaccia vocale naturale e intuitiva, sfruttando tecniche avanzate di ASR e un'integrazione fluida con Unity.

---

## **Scopo del Progetto**
Lo scopo è sviluppare un sistema ASR in grado di riconoscere **14 comandi vocali specifici** e utilizzarli per controllare un avatar, migliorando l'accessibilità e l'interattività in un ambiente virtuale.

---

## **Struttura del Progetto**
Il progetto è organizzato come segue:

- **Dataset**: File audio personalizzati per il training.
- **Manifest JSON**: File per la gestione dei dati audio durante il training e la validazione.
- **Script Python**: Preprocessing, addestramento e ottimizzazione del modello
- **Modelli**: Modelli ASR addestrati e ottimizzati (es. `asr_model.nemo`).
- **Server ASR**: Un servizio REST basato su Flask per eseguire inferenze vocali.
- **Unity Client**: Script C# per catturare i comandi vocali e tradurli in azioni.

---

## **Comandi Supportati**
Il sistema riconosce i seguenti comandi vocali:

- **"avanti"**: Sposta l'avatar in avanti.
- **"indietro"**: Sposta l'avatar all'indietro.
- **"sinistra"**: Ruota l'avatar a sinistra.
- **"destra"**: Ruota l'avatar a destra.
- **"salta"**: Fa saltare l'avatar.
- **"vola"**: Attiva la modalità di volo.
- **"su"**: Sposta l'avatar in alto.
- **"giù"**: Sposta l'avatar in basso.
- **"fermo"**: Ferma tutti i movimenti dell'avatar.
- **"corri"**: Attiva la modalità di corsa.
- **"cammina"**: Attiva la camminata normale.
- **"pausa"**: Mette il gioco in pausa.
- **"continua"**: Riprende il gioco dopo la pausa.
- **"esci"**: Chiude l'applicazione Unity.

---

## **Requisiti di Sistema**
- **Python 3.8+**
- **Unity 2021+**
- **NVIDIA NeMo** e **PyTorch**
- Microfono per la registrazione dei comandi vocali

---

## **Installazione**

### **1. Clona il Repository**
```bash
git clone https://github.com/Giosc4/NLP_Project.git
cd NLP_Project
```

### **2. Configura l'Ambiente**

Se utilizzi **Windows Subsystem for Linux (WSL)** come ambiente, crea un ambiente virtuale e installa le librerie necessarie:

1. Crea e attiva un ambiente virtuale:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Su Windows: venv\Scripts\activate
   ```

2. Installa le dipendenze:
   ```bash
   pip install -r [library]
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

---

### **3. Avvia il Server ASR**

Il server ASR è il componente che gestisce le inferenze vocali. Avvialo con:
```bash
python asr_server.py
```
Il server sarà disponibile all'indirizzo `http://localhost:5001`.

Questo server aspetta input audio dal client Unity e restituisce una predizione basata sui comandi riconosciuti.

---

## **Configurazione di Unity**
1. Apri il progetto Unity disponibile nella cartella `UnityNLP_Script`.
2. Aggiungi lo script `VoiceCommandHandler.cs` al GameObject principale.
3. Configura l'URL API per il server ASR:
   ```csharp
   public string apiURL = "http://localhost:5001/predict";
   ```
4. Premi **Play** e utilizza il tasto `V` per registrare i comandi vocali.
   Unity invierà le registrazioni al server, che risponderà con la predizione del comando da eseguire.

---

## **Testing del Sistema**
Durante la fase di testing, i comandi vocali inviati vengono salvati e possono essere riutilizzati per migliorare il modello.

---

## **Risultati**
- **Accuratezza**: 87%
- **Word Error Rate (WER)**: 13%
- **Character Error Rate (CER)**: 9%
- **Performance**: Controllo fluido e reattivo dell'avatar.

---

## **Possibili Sviluppi Futuri**
- Ampliamento del dataset.
- Ampliamento delle applicazioni 3d utilizzabili.
- Introduzione di nuovi comandi vocali e ottimizzazione delle animazioni.

---

## **Crediti**
- **NVIDIA NeMo**: Framework ASR.
- **PyTorch**: Backend per l'addestramento.
- **Unity**: Ambiente grafico.
- **Optuna**: Ottimizzazione degli iperparametri.
- **ChatGPT** e **Windsurf AI**: Supporto nella progettazione e sviluppo.
- **Fabio Tamburini**: Per il supporto nel corso e nello sviluppo del modello.

---

## **Riferimenti**
- **NVIDIA NeMo**: [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
- **PyTorch**: [https://pytorch.org/](https://pytorch.org/)
- **Unity**: [https://docs.unity3d.com/](https://docs.unity3d.com/)
- **Optuna**: [https://optuna.org/](https://optuna.org/)

---

Se hai domande o suggerimenti, sentiti libero di aprire un'**issue** o contattarmi!
