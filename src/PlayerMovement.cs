using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

// PlayerMovement.cs
public class PlayerMovement : MonoBehaviour
{
    public CharacterController controller;

    public float speed = 100f;
    public float walkSpeed = 10f;
    public float runSpeed = 18f;
    public float gravity = -9.81f * 2;
    public float jumpHeight = 3f;

    // Booleane per i comandi vocali
    private bool moveForwardVoice = false;
    private bool moveBackwardVoice = false;
    private bool moveLeftVoice = false;
    private bool moveRightVoice = false;
    private bool moveUpVoice = false;   // per volare più in alto quando vola
    private bool moveDownVoice = false; // per volare più in basso quando vola

    private bool isRunning = false; // Indica se l'avatar sta correndo

    public bool isFlying = false;
    private bool initialFlyBoostApplied = false; // Per controllare se la spinta iniziale è stata applicata

    private float xRotation = 0f; // Rotazione sull'asse X (se necessario in futuro)
    private float yRotation = 0f; // Rotazione sull'asse Y (per destra/sinistra)
    public float rotationAmount = 35f; // Valore regolabile della rotazione in gradi

    public Transform cameraTransform;
    public Transform groundCheck;
    public float groundDistance = 0.2f; // Aumentato per un check più ampio
    public LayerMask groundMask;

    Vector3 velocity;
    bool isGrounded;

    void Update()
    {
        // Controllo se siamo a terra
        isGrounded = Physics.CheckSphere(groundCheck.position, groundDistance, groundMask);

        // Se non stiamo volando, applichiamo gravità normale
        if (!isFlying)
        {
            if (isGrounded && velocity.y < 0)
            {
                velocity.y = -2f; // Resetta la velocità verticale quando a terra
            }

            // Gravità continua se non voliamo
            velocity.y += gravity * Time.deltaTime;
        }
        else
        {
            // Se stiamo volando e la spinta iniziale è stata applicata, manteniamo la quota
            if (initialFlyBoostApplied)
            {
                velocity.y = Mathf.Lerp(velocity.y, 0f, Time.deltaTime * 5f); // Stabilizzazione graduale
            }
        }

        // Se la corsa automatica è attiva, il personaggio continua a muoversi in avanti
        if (isRunning && !isFlying)
        {
            controller.Move(transform.forward * speed * Time.deltaTime);
        }

        // Movimento basato sui comandi vocali
        Vector3 move = Vector3.zero;

        if (moveForwardVoice) move += transform.forward;
        if (moveBackwardVoice) move -= transform.forward;

        if (moveLeftVoice)
        {
            move -= transform.right;
        }

        if (moveRightVoice)
        {
            move += transform.right;
        }

        // Se stiamo volando e i comandi vocali su/giu sono attivi
        if (isFlying)
        {
            if (moveUpVoice) move += transform.up;
            if (moveDownVoice)
            {
                move -= transform.up;

                // Se tocca terra mentre il comando giù è attivo, interrompe il volo
                if (isGrounded)
                {
                    isFlying = false;
                    initialFlyBoostApplied = false; // Resetta la spinta iniziale
                    StopVerticalMovement();
                    Debug.Log("Interrotto il volo: il personaggio è atterrato.");
                }
            }
        }

        // Se la corsa automatica è attiva, il personaggio continua a muoversi in avanti
        if (isRunning && !isFlying)
        {
            controller.Move(transform.forward * runSpeed * Time.deltaTime);
        }

        // Applica il movimento orizzontale/verticale
        controller.Move(speed * Time.deltaTime * move);

        // Applica la gravità
        controller.Move(velocity * Time.deltaTime);
    }

    public void MoveForward()
    {
        StopAllCoroutines(); // Interrompe eventuali movimenti attivi
        StartCoroutine(MoveForwardForDistance());
    }

    public void MoveBackward()
    {
        moveForwardVoice = false;  // Interrompe il movimento in avanti
        isRunning = false;         // Disattiva la corsa
        moveBackwardVoice = true;  // Attiva il movimento indietro
        speed = walkSpeed;         // Imposta la velocità di camminata
        Debug.Log("Il personaggio sta andando indietro.");
    }

    private IEnumerator MoveForwardForDistance()
    {
        ResetMovement(); // Resetta gli altri movimenti

        Vector3 startPosition = transform.position;
        Vector3 targetPosition = startPosition + transform.forward * walkSpeed;

        while (Vector3.Distance(transform.position, targetPosition) > 0.1f)
        {
            // Muovi il personaggio verso la destinazione
            Vector3 moveDirection = (targetPosition - transform.position).normalized;
            controller.Move(moveDirection * walkSpeed * Time.deltaTime);

            yield return null; // Aspetta il frame successivo
        }

        Debug.Log("Movimento avanti completato.");
    }

    public void MoveRight()
    {
        transform.Rotate(Vector3.up * rotationAmount);
        Debug.Log("Rotazione a destra di " + rotationAmount + " gradi applicata.");
    }

    public void MoveLeft()
    {
        transform.Rotate(Vector3.up * -rotationAmount);
        Debug.Log("Rotazione a sinistra di " + rotationAmount + " gradi applicata.");
    }

    private void ApplyRotation()
    {
        // Invece di ruotare il giocatore, ruota direttamente la camera
        cameraTransform.localRotation = Quaternion.Euler(xRotation, yRotation, 0f);
    }

    public void MoveUp() // quando vola
    {
        moveUpVoice = true;
        moveDownVoice = false;
    }

    public void MoveDown() // quando vola
    {
        moveDownVoice = true;
        moveUpVoice = false;
    }

    public void LookUp()
    {
        xRotation -= rotationAmount; // Riduci l'angolo per guardare in alto
        xRotation = Mathf.Clamp(xRotation, -90f, 90f); // Limita l'angolo tra -90 e 90°
        ApplyRotation();
    }

    public void LookDown()
    {
        xRotation += rotationAmount; // Aumenta l'angolo per guardare in basso
        xRotation = Mathf.Clamp(xRotation, -90f, 90f); // Limita l'angolo tra -90° e 90°
        ApplyRotation();
    }

    public void StopVerticalMovement()
    {
        moveUpVoice = false;
        moveDownVoice = false;
    }

    public void StopMovement()
    {
        // Ferma tutti i movimenti orizzontali
        moveForwardVoice = false;
        moveBackwardVoice = false;
        moveLeftVoice = false;
        moveRightVoice = false;

        // Ferma corsa e camminata
        isRunning = false;

        // Se stiamo volando, fermiamo anche i movimenti verticali
        if (isFlying)
        {
            moveUpVoice = false;
            moveDownVoice = false;
            Debug.Log("Movimento verticale fermato durante il volo.");
        }

        Debug.Log("Tutti i movimenti sono stati fermati.");
    }

    public void JumpAction()
    {
            // Aggiunge solo la forza del salto sulla Y mantenendo il movimento orizzontale esistente
            velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
            Debug.Log("Salto eseguito!");
        
    }

    public void Run()
    {
        moveBackwardVoice = false; // Interrompe il movimento indietro
        moveForwardVoice = true;   // Attiva il movimento in avanti
        isRunning = true;          // Attiva lo stato di corsa
        speed = runSpeed;          // Imposta la velocità di corsa
        Debug.Log("Il personaggio sta correndo.");
    }

    public void Walk()
    {
        moveBackwardVoice = false;  // Interrompe il movimento all'indietro
        moveForwardVoice = true;   // Attiva il movimento in avanti
        isRunning = false;  // Disattiva la corsa
        speed = walkSpeed;  // Imposta la velocità di camminata
        Debug.Log("Il personaggio sta camminando.");
    }

    public void Fly()
    {
        // Attiva il volo
        isFlying = true;

        if (!initialFlyBoostApplied)
        {
            float initialFlyBoost = 20f; // Altezza iniziale regolabile
            velocity.y = Mathf.Sqrt(initialFlyBoost * -2f * gravity); // Calcola l'impulso iniziale per salire
            initialFlyBoostApplied = true; // Segnala che la spinta iniziale è stata applicata
            Debug.Log("Il volo è stato attivato: il personaggio è salito di " + initialFlyBoost + " unità.");
        }
    }

    private void ResetMovement()
    {
        // Se vuoi che ogni nuovo comando resetti i precedenti
        moveForwardVoice = false;
        moveBackwardVoice = false;
        moveLeftVoice = false;
        moveRightVoice = false;
    }
}
