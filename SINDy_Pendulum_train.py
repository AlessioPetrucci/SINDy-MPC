
import numpy as np
import pandas as pd
import pysindy as ps
from sklearn.utils import resample
import gymnasium as gym
import pickle # per STLSQ


# --- Configurazione Pendolo ---
N_ENSEMBLE = 20        # Numero di modelli per quantificare l'incertezza
TOTAL_ITERS = 1000     # Iterazioni totali di training
UPDATE_FREQ = 20       # Ogni quante iterazioni rifacciamo il fit
LIBRARY_DEGREE = 2     # Grado dei polinomi (cos e sin sono già negli stati)
dt = 0.05 # Passo temporale standard di Pendulum-v1

def pendulum_expert(obs):
    
    #Controllore Proporzionale-Derivativo (PD) per guidare il pendolo verso l'alto.
    #Stato obs: [cos(theta), sin(theta), theta_dot]
    
    cos_th, sin_th, th_dot = obs
    theta = np.arctan2(sin_th, cos_th) # Calcolo dell'angolo reale
    
    # Guadagni del controllore (K_p e K_d)
    kp = 8.0
    kd = 0.5
    
    # Calcolo della coppia (torque): cerca di annullare angolo e velocità angolare
    u = -kp * theta - kd * th_dot
    return np.array([np.clip(u, -2.0, 2.0)]) # Clip al range dell'ambiente [-2, 2]

def select_best_performer(ensemble, env):
    
    #Valuta ogni modello dell'ensemble su un episodio di validazione reale.
    
    print("\n--- Fase di Selezione Best Performer (Pendolo) ---")
    
    # 1. Raccogliamo un episodio di validazione "fresco"
    obs, _ = env.reset()
    val_data = []
    done = False
    for _ in range(200): # Test su 2 secondi di simulazione
        action = pendulum_expert(obs)
        next_obs, _, term, trunc, _ = env.step(action)
        val_data.append((obs, action, next_obs))
        obs = next_obs
        if term or trunc: break
    
    best_model = None
    min_mae = float('inf')
    
    # 2. Testiamo ogni modello dell'ensemble sulla predizione a 1-step
    for idx, model in enumerate(ensemble):
        errors = []
        for s, a, s_next in val_data:
            pred = model.predict(s.reshape(1, -1), u=a.reshape(1, -1))[0]
            err = np.mean(np.abs(pred - s_next)) # Errore su [cos, sin, v]
            errors.append(err)
        
        current_mae = np.mean(errors)
        if current_mae < min_mae:
            min_mae = current_mae
            best_model = model
            best_idx = idx
            
    print(f"Vincitore: Modello {best_idx} con MAE di validazione: {min_mae:.6f}")
    return best_model

def train_ensemble(data):
    #Apprende un ensemble di modelli SINDy per il pendolo
    models = []
    # Indici Pendolo: 3 stati (0,1,2), 1 azione (3), 3 target (4,5,6)
    X = data.iloc[:, :3].values  # [cos, sin, v]
    U = data.iloc[:, 3:4].values # [torque]
    Y = data.iloc[:, 4:].values  # [cos_next, sin_next, v_next]
    
    feature_library = ps.PolynomialLibrary(degree=LIBRARY_DEGREE)


    for i in range(N_ENSEMBLE):
        # Bootstrap per diversificare i modelli
        X_res, U_res, Y_res = resample(X, U, Y, random_state=i)
        
        model = ps.SINDy(
            feature_library=feature_library,
            optimizer=ps.STLSQ(threshold=0.001), # Soglia per la parsimonia
            feature_names=['cos', 'sin', 'v', 'u']
        )
        
    
        model.fit([X_res], u=[U_res], x_dot=[Y_res], quiet=True, multiple_trajectories = True)      
        models.append(model)
    return models

def run_sindy_pendulum():
    # 1. Caricamento dati iniziali (devono avere 7 colonne)
    datastore = pd.read_csv("pendulum_data.csv")
    env = gym.make("Pendulum-v1", render_mode=None)
    
    ensemble = []
    for i in range(TOTAL_ITERS):
        if i % UPDATE_FREQ == 0:
            print(f"Iterazione {i}: Update Model e Raccolta Dati Esperte...")
            ensemble = train_ensemble(datastore)
            
            # 2. Generazione Nuove Traiettorie (Refit verso la verticale)
            new_data = []
            for _ in range(5): # 5 tentativi di salita partendo da punti diversi
                seed_val = np.random.randint(0, 99999)
                obs, _ = env.reset(seed=seed_val)


                # valori per gli input sinusoidali
                amplitude = np.random.uniform(1.0, 2.0) 
                freq = np.random.uniform(0.5, 3.0) # Frequenza in Hz
                phase = np.random.uniform(0, np.pi) 
                
                done = False
                for t_step in range(200): # 5 secondo di controllo esperto
                    

                    
                    #imput sinusoidali e pd esperto 
                    time = t_step * dt
                    u_val = amplitude * np.sin(2 * np.pi * freq * time + phase)
                    action = np.array([np.clip(u_val, -2.0, 2.0)]) # Clip di sicurezza
                    


                    """ 
                    #input sinusoidali random
                    expert_u = pendulum_expert(obs)[0]
                    
                    # Perturbazione sinusoidale: u_p(t) = A \cdot \sin(\omega t)
                    time = t_step * dt
                    sin_noise = amplitude * np.sin(2 * np.pi * freq * time)
                    
                    # Azione finale: Esperto + Disturbo, con Clip di sicurezza [-2, 2]
                    action = np.array([np.clip(expert_u + sin_noise, -2.0, 2.0)])
                    """

                    next_obs, _, term, trunc, _ = env.step(action)
                    
                    # Salvataggio transizione completa
                    new_data.append(np.concatenate([obs, action, next_obs]))
                    
                    obs = next_obs
                    if term or trunc: break
            
            # Aggiornamento del Datastore
            new_df = pd.DataFrame(new_data, columns=datastore.columns)
            datastore = pd.concat([datastore, new_df], ignore_index=True)
            print(f"Aggiunti {len(new_data)} campioni. Totale: {len(datastore)}")

    # 3. Selezione del Best Performer finale
    final_model = select_best_performer(ensemble, env)

    # 4. Salvataggio del modello
    filename = "sindy_pendulum_sinusSTLSQ.pkl"
    with open(filename, "wb") as f:
        pickle.dump(final_model, f)
    
    print(f"\nModello Pendolo salvato: {filename}")
    print("Equazioni Identificate:")
    final_model.print()

if __name__ == "__main__":
    run_sindy_pendulum()
