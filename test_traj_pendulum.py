
import numpy as np
import pandas as pd
import pickle  # per STLSQ
#import dill as pickle
import gymnasium as gym
import matplotlib.pyplot as plt

def run_pendulum_comparison(model_path, steps=200):
    # 1. Caricamento Modello e Ambiente
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    env = gym.make("Pendulum-v1", render_mode=None)

    obs, _ = env.reset(seed=42) 
    
    obs_real = obs.copy()
    obs_sindy = obs.copy()
    
    history_real = []
    history_sindy = []
    
    # 2. Rollout Ricorsivo con Vincolo Fisico
    print(f"Esecuzione rollout di {steps} passi...")
    for t in range(steps):
        u = np.array([2.0 * np.sin(t * 0.1)]) 
        
        # Step Reale
        history_real.append(obs_real)
        obs_real, _, _, _, _ = env.step(u)
        
        # Step SINDy con correzione
        history_sindy.append(obs_sindy)
        
        try:
            pred = model.predict(obs_sindy.reshape(1, -1), u=u.reshape(1, -1))[0]
            
            # VINCOLO FISICO: Manteniamo la punta sulla circonferenza unitaria
            cos_val, sin_val, v_val = pred
            norm = np.sqrt(cos_val**2 + sin_val**2)
            obs_sindy = np.array([cos_val/norm, sin_val/norm, v_val])
            
            if np.any(np.abs(obs_sindy) > 1e6):
                break
        except ValueError:
            break

    history_real = np.array(history_real)
    history_sindy = np.array(history_sindy)
    t_axis = np.arange(len(history_real)) * 0.05

    # --- GRAFICO 1: Predizioni nel tempo (Invariato) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    state_labels = [r'$\cos(\theta)$', r'$\sin(\theta)$', r'$\dot{\theta}$ (Velocità)']
    colors = ['b', 'g', 'r']

    for i in range(3):
        axes[i].plot(t_axis, history_real[:, i], color=colors[i], label='Reale', alpha=0.4)
        axes[i].plot(t_axis, history_sindy[:, i], color='black', linestyle='--', label='SINDy')
        axes[i].set_ylabel(state_labels[i])
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
    axes[2].set_xlabel("Tempo [s]")
    plt.suptitle("Confronto Dinamico: Pendolo Reale vs SINDy")
    plt.tight_layout()
    #plt.savefig("state_comparison_pendulum", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_pendulum_comparison("sindy_pendulum_sinusfissoSTLSQ.pkl")
    