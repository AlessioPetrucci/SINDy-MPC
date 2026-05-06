# Data-Driven Control of an Inverted Pendulum via SINDy and MPC

[![Python 3.10.12](https://img.shields.io/badge/python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-31012/)

Questo repository implementa un framework completo per l'**identificazione sistematica della dinamica** di un pendolo invertito e il suo successivo **controllo ottimo**. L'approccio combina l'algoritmo *Sparse Identification of Nonlinear Dynamics* (**SINDy**) per la creazione di un Digital Twin e un *Model Predictive Control* (**MPC**) per la stabilizzazione dello stato.

## 🎯 Obiettivi del Progetto
L'obiettivo primario è dimostrare che è possibile controllare un sistema non lineare senza conoscerne a priori le equazioni fisiche (es. equazioni di Eulero-Lagrange). Il sistema "scopre" la struttura matematica sottostante direttamente dai dati osservati e la utilizza come modello predittivo per l'ottimizzazione del controllo.

---

## 🏛️ Architettura del Sistema

Il workflow è suddiviso in quattro moduli funzionali indipendenti:

### 1. Acquisizione Dati (`pendulum_data.py`)
Generazione di un dataset sintetico iniziale tramite esplorazione stocastica dell'ambiente `Gymnasium Pendulum-v1`.
* **Input**: Azioni casuali nel range di torque $[-2.0, 2.0]$.
* **Output**: Dataset in formato CSV (`pendulum_data.csv`) contenente lo stato $x = [\cos\theta, \sin\theta, \dot{\theta}]$ e le relative transizioni.

### 2. Identificazione della Dinamica (`SINDy_Pendulum_train.py`)
Fase di addestramento basata su regressione sparsa utilizzando la libreria `PySINDy`.
* **Ensemble Learning**: Utilizzo di 20 modelli SINDy per la quantificazione dell'incertezza e la robustezza statistica.
* **Active Sampling**: Implementazione di una raccolta dati "esperta" mediante controller PD e disturbi sinusoidali per campionare lo spazio degli stati in prossimità del punto di equilibrio instabile.
* **Model Selection**: Selezione del *Best Performer* basata sulla minimizzazione del Mean Absolute Error (MAE) su un episodio di validazione.

### 3. Validazione e Manifold Projection (`test_traj_pendulum.py`)
Analisi della capacità di generalizzazione del modello identificato.
* **Recursive Rollout**: Confronto tra la traiettoria reale (fisica di Gymnasium) e quella predetta dal modello SINDy.
* **Geometric Constraint**: Applicazione di una proiezione sul manifold ($\cos^2\theta + \sin^2\theta = 1$) per garantire la coerenza trigonometrica dello stato predetto ed evitare divergenze numeriche.

### 4. Controllo Predittivo (`Pendulum_MPC.py`)
Implementazione del controllore a orizzonte recedente.
* **Ottimizzazione**: Minimizzazione di una funzione di costo quadratica su un orizzonte temporale di 15 step.
* **Solver**: Utilizzo dell'algoritmo `L-BFGS-B` per la ricerca della sequenza di input ottimale tra $[-2.0, 2.0]$.
* **Target**: Stabilizzazione al punto di equilibrio $[1, 0, 0]$ (pendolo in posizione verticale).

---

## 🛠️ Requisiti Installazione

Le dipendenze necessarie possono essere installate tramite `pip`:

```bash
pip install gymnasium pysindy scipy matplotlib tqdm pandas numpy dill
