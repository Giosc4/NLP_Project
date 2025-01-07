using UnityEngine;
using System.Collections;
using UnityEngine.Networking;
using System.Text;
using System.IO;
using System.Net;
using System.Collections.Generic;

public class VoiceCommandHandler : MonoBehaviour
{
    public string apiURL = "http://localhost:5001/predict";
    private string microphoneName;
    private int sampleRate = 16000; // 16kHz
    private AudioClip microphoneClip;
    public PlayerMovement playerMovement;
    public MouseMovement mouseMovement;
    private AudioClip recordingClip; // Clip per accumulare l'audio
    private List<float> recordedSamples = new List<float>(); // Campioni audio accumulati.
    private bool isGamePaused = false; // Controlla se il gioco è in pausa
    public GameObject pauseCanvas; // Assegna il canvas dalla scena nel tuo inspector


    private float recordingStartTime;
    private float minRecordingTime = 1.0f;

    void Start()
    {
        if (Microphone.devices.Length > 0)
        {
            microphoneName = Microphone.devices[0];
        }
        else
        {
            Debug.LogError("Nessun microfono trovato!");
        }

        playerMovement = FindObjectOfType<PlayerMovement>();
        mouseMovement = FindObjectOfType<MouseMovement>();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.V))
        {
            StartRecording();
        }

        if (Input.GetKeyUp(KeyCode.V))
        {
            StopRecording();
        }
    }

    void StartRecording()
    {
        recordedSamples.Clear(); // Resetta i campioni registrati
        recordingClip = Microphone.Start(microphoneName, true, 5, sampleRate); // Registrazione continua
        recordingStartTime = Time.time; // Registra il tempo di inizio
        Debug.Log("Registrazione iniziata...");
    }

    void StopRecording()
    {
        StartCoroutine(EnsureMinRecordingTime());

    }

    IEnumerator EnsureMinRecordingTime()
    {
        float elapsedTime = Time.realtimeSinceStartup - recordingStartTime;
        float remainingTime = minRecordingTime - elapsedTime;

        if (remainingTime > 0)
        {
            // Usa WaitForSecondsRealtime per attendere in modo indipendente da timeScale
            yield return new WaitForSecondsRealtime(remainingTime);
        }

        int sampleCount = Microphone.GetPosition(microphoneName);
        Microphone.End(microphoneName);

        if (sampleCount <= 0)
        {
            Debug.LogWarning("Registrazione troppo breve, campioni insufficienti.");
            yield break;
        }

        float[] samples = new float[sampleCount * recordingClip.channels];
        recordingClip.GetData(samples, 0);
        recordedSamples.AddRange(samples);

        Debug.Log("Registrazione terminata. Invio dell'audio...");
        SendRecordedAudio();
    }

    void SendRecordedAudio()
    {
        AudioClip finalClip = AudioClip.Create("FinalClip", recordedSamples.Count, recordingClip.channels, sampleRate, false);
        finalClip.SetData(recordedSamples.ToArray(), 0);

        StartCoroutine(SendAudioClip(finalClip));
    }

    IEnumerator SendAudioClip(AudioClip clip)
    {
        byte[] wavData = WavUtility.FromAudioClip(clip);
        WWWForm form = new WWWForm();
        form.AddBinaryData("file", wavData, "audio.wav", "audio/wav");

        UnityWebRequest request = UnityWebRequest.Post(apiURL, form);

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonResponse = request.downloadHandler.text;
            Debug.Log("Risposta dall'API: " + jsonResponse);
            ProcessCommand(jsonResponse);
        }
        else
        {
            Debug.LogError($"Errore nella richiesta: {request.error}");
        }
    }

    void ProcessCommand(string jsonResponse)
    {
        CommandResponse response = JsonUtility.FromJson<CommandResponse>(jsonResponse);
        string command = response.command.ToLower();

        // Se il gioco è in pausa, ignora tutti i comandi tranne "continua" e "esci"
        if (isGamePaused && command != "continua" && command != "esci")
        {
            Debug.Log("Il gioco è in pausa. Comando ignorato: " + command);
            return;
        }

        // Se stiamo volando o no influenza il comportamento di "su" e "giu"
        bool currentlyFlying = playerMovement.isFlying;

        switch (command)
        {
            case "avanti":
                playerMovement.MoveForward();
                break;
            case "indietro":
                playerMovement.MoveBackward();
                break;
            case "sinistra":
                playerMovement.MoveLeft();
                break;
            case "destra":
                playerMovement.MoveRight();
                break;
            case "salta":
                playerMovement.JumpAction();
                break;
            case "fermo":
                playerMovement.StopMovement();
                break;
            case "pausa":
                PauseGame();
                break;
            case "continua":
                ResumeGame();
                break;
            case "esci":
                ExitGame();
                break;
            case "corri":
                playerMovement.Run();
                break;
            case "cammina":
                playerMovement.Walk();
                break;
            case "su":
                if (currentlyFlying)
                {
                    // Se stiamo volando, ci muoviamo verso l'alto
                    playerMovement.MoveUp();
                }
                else
                {
                    // Se non stiamo volando, guardiamo verso l'alto
                    playerMovement.LookUp();
                }
                break;

            case "giu":
                if (currentlyFlying)
                {
                    // Se stiamo volando, ci muoviamo verso il basso
                    playerMovement.MoveDown();
                }
                else
                {
                    // Se non stiamo volando, guardiamo verso il basso
                    playerMovement.LookDown();
                }
                break;
            case "vola":
                playerMovement.Fly();
                break;

            default:
                Debug.Log("Comando non riconosciuto: " + command);
                break;
        }
    }

    void PauseGame()
    {
        isGamePaused = true;
        Time.timeScale = 0f; // Ferma il gioco
        Debug.Log("Gioco in pausa.");
    }

    void ResumeGame()
    {
        isGamePaused = false;
        Time.timeScale = 1f; // Riprendi il gioco
        Debug.Log("Gioco ripreso.");
    }

    public void ExitGame()
    {
        Debug.Log("Uscita dal gioco.");
        Application.Quit();
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
    }

    void OnGUI()
    {
        if (isGamePaused)
        {
            GUIStyle style = new GUIStyle(GUI.skin.label);
            style.fontSize = 40;
            style.alignment = TextAnchor.MiddleCenter;
            style.normal.textColor = Color.white;

            // Posizioniamo il testo al centro dello schermo
            Rect rect = new Rect(0, 0, Screen.width, Screen.height);
            GUI.Label(rect, "Gioco in Pausa", style);
        }
    }
}

[System.Serializable]
public class CommandResponse
{
    public string command;
}
