import gymnasium as gym
import numpy as np
import pandas as pd

def generate_pendulum_data(filename="pendulum_data.csv", steps=5000):
    env = gym.make("Pendulum-v1", render_mode=None)
    obs, _ = env.reset()
    
    data = []
    for _ in range(steps):
        # Azione casuale (torque tra -2.0 e 2.0)
        action = env.action_space.sample() 
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Formato: [cos(th), sin(th), d_th, torque, cos(th)_next, sin(th)_next, d_th_next]
        data.append(np.concatenate([obs, action, next_obs]))
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    cols = ['s0', 's1', 's2', 'a0', 's0_next', 's1_next', 's2_next']
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(filename, index=False)
    print(f"Dataset pendolo salvato: {len(df)} campioni.")

if __name__ == "__main__":
    generate_pendulum_data()
