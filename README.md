# SINDy-MPC
Questo progetto dimostra come utilizzare l'algoritmo SINDy (Sparse Identification of Nonlinear Dynamics) per creare un gemello digitale (Digital Twin) di un pendolo e controllarlo tramite MPC (Model Predictive Control).

Il progetto si articola in quattro fasi principali, ognuna gestita da un file specifico:pendulum_data.py: Generazione Dataset.Esegue azioni casuali nell'ambiente Pendulum-v1.  Salva le transizioni $[cos(\theta), sin(\theta), \dot{\theta}, \tau]$ in un file CSV per l'addestramento iniziale.  SINDy_Pendulum_train.py: Addestramento e Identificazione.Utilizza un ensemble di 20 modelli SINDy per gestire l'incertezza.  Implementa una "raccolta dati esperta" usando un controller PD per esplorare meglio le zone vicine alla posizione verticale.  Salva il miglior modello trovato (Best Performer) in un file .pkl.  test_traj_pendulum.py: Validazione del Modello.Esegue un test di confronto tra la fisica reale e la predizione del modello SINDy.  Applica un vincolo fisico (Manifold Projection) per assicurarsi che i valori di seno e coseno siano coerenti tra loro.  Pendulum_MPC.py: Controllo Predittivo.Implementa il loop di controllo MPC.  Ad ogni passo, risolve un problema di ottimizzazione su un orizzonte di 15 step per trovare la coppia ($torque$) ideale che minimizza il costo rispetto al target verticale.


Requisiti

Per far girare il progetto, assicurati di avere installato:

gymnasium: Per l'ambiente di simulazione.  

pysindy: Per l'identificazione delle equazioni.  

scipy: Per l'ottimizzazione dell'MPC (metodo L-BFGS-B).  

matplotlib & tqdm: Per la visualizzazione dei risultati e il monitoraggio del progresso.

Risultati 

Il sistema è in grado di:Identificare le equazioni: SINDy stampa a video le equazioni identificate (es. la relazione tra velocità e accelerazione angolare).  Controllare il sistema: L'MPC stabilizza il pendolo nella posizione $cos(\theta) = 1$ anche se il modello è appreso puramente dai dati.
