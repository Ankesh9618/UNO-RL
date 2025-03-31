import gym
import numpy as np
import torch
from uno_env import UnoEnv  # Make sure this matches the actual environment class name
from model import DQN  # Make sure this matches the actual model class name

def evaluate_model(env, model, episodes=100):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state
        rewards.append(total_reward)
        print(f"Episode {episode + 1}/{episodes} - Reward: {total_reward}")
    avg_reward = np.mean(rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    # Load the environment
    env = UnoEnv()

    # Load the trained model
    model_path = "model.pth"  # Adjust the path to your trained model
    model = DQN(env.observation_space.shape[0], env.action_space.n)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate the model
    evaluate_model(env, model)