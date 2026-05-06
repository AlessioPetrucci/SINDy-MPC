import numpy as np
import gymnasium as gym
import dill as pickle
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configurazione MPC ---
HORIZON = 15        
DT = 0.05           
Q = np.diag([10.0, 10.0, 0.1]) 
R = 0.01            
TARGET = np.array([1.0, 0.0, 0.0]) # cos=1, sin=0, vel=0

def manifold_projection(state):
    """Proiezione geometrica per mantenere la coerenza fisica."""
    cos_th, sin_th, v = state
    norm = np.sqrt(cos_th**2 + sin_th**2)
    return np.array([cos_th/norm, sin_th/norm, v])

def mpc_cost_function(u_sequence, current_state, model, horizon):
    cost = 0
    state = current_state.copy()
    u_sequence = u_sequence.reshape(horizon, 1)
    for i in range(horizon):
        u = u_sequence[i]
        # Predizione nel futuro tramite il Digital Twin SINDy
        state = model.predict(state.reshape(1, -1), u=u.reshape(1, -1))[0]
        state = manifold_projection(state)
        state_error = state - TARGET
        cost += state_error.T @ Q @ state_error + R * (u**2)
    return cost


def run_comparison(model_path, steps=40):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Inizializzazione Ambiente Gymnasium (Senza rendering per velocità)
    env = gym.make("Pendulum-v1", render_mode=None) 
    obs_real, _ = env.reset(seed=42)
    
    # Inizializzazione Mondo Surrogato
    obs_surr = obs_real.copy() 

    history_real, history_surr = [], []
    u_real, u_surr = [], []

    # Aggiungiamo tqdm qui per monitorare il progresso
    print(f"Inizio simulazione MPC ({steps} passi)...")
    for t in tqdm(range(steps), desc="Calcolo Traiettorie MPC", unit="step"):
        
        # 1. Ottimizzazione Receding Horizon per il mondo Reale
        res_real = minimize(mpc_cost_function, np.zeros(HORIZON), args=(obs_real, model, HORIZON),
                            method='L-BFGS-B', bounds=[(-2.0, 2.0)]*HORIZON)
        action_real = np.array([res_real.x[0]])
        
        # 2. Ottimizzazione Receding Horizon per il mondo Surrogato
        res_surr = minimize(mpc_cost_function, np.zeros(HORIZON), args=(obs_surr, model, HORIZON),
                            method='L-BFGS-B', bounds=[(-2.0, 2.0)]*HORIZON)
        action_surr = np.array([res_surr.x[0]])

        # Salvataggio dati
        history_real.append(obs_real)
        history_surr.append(obs_surr)
        u_real.append(action_real)
        u_surr.append(action_surr)

        # Update degli stati
        obs_real, _, _, _, _ = env.step(action_real)
        
        # Predizione puramente matematica tramite SINDy
        obs_surr = model.predict(obs_surr.reshape(1, -1), u=action_surr.reshape(1, -1))[0]
        obs_surr = manifold_projection(obs_surr) # Vincolo geometrico

    env.close()
    return (np.array(history_real), np.array(history_surr), 
            np.array(u_real), np.array(u_surr))


# --- Plotting dei risultati ---
def plot_results(h_real, h_surr, u_real, u_surr):
    t = np.arange(len(h_real)) * DT
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    labels = [r'$\cos(\theta)$', r'$\sin(\theta)$', r'Velocità $\dot{\theta}$', 'Input $u$ (Torque)']
    colors = ['blue', 'red']
    
    for i in range(3):
        axs[i].plot(t, h_real[:, i], color=colors[0], label='Gym (Reale)')
        axs[i].plot(t, h_surr[:, i], color=colors[1], linestyle='--', label='SINDy (Surrogato)')
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
        if i == 0: axs[i].legend(loc='upper right')

    # Plot degli Input
    axs[3].plot(t, u_real.flatten(), color=colors[0], label='u Reale')
    axs[3].plot(t, u_surr.flatten(), color=colors[1], linestyle='--', label='u Surrogato')
    axs[3].set_ylabel(labels[3])
    axs[3].set_xlabel("Tempo [s]")
    axs[3].grid(True, alpha=0.3)
    
    plt.suptitle("Confronto MPC: Sistema Reale vs Modello SINDy")
    plt.tight_layout()
    #plt.savefig("pendulum_MPC_comparison", dpi = 300)
    plt.show()

if __name__ == "__main__":
    h_r, h_s, u_r, u_s = run_comparison("sindy_pendulum_best.pkl")
    plot_results(h_r, h_s, u_r, u_s)