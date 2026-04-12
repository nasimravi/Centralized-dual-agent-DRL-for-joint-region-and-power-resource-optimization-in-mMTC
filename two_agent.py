import signal
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from tqdm import tqdm
from time import time
from env_2agent_constant import env_2agent
try:
    plt.style.use('seaborn-v0_8-darkgrid')  # Newer matplotlib versions
except:
    plt.style.use('dark_background')  # Fallback option
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQNAgent:
    def __init__(self, state_size, action_size, agent_id):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model().to(device)
        self.target_update_freq = 50  # New attribute
        self.target_model = self._build_model().to(device)
        self.update_target_model()  # Now properly defined below




    def _build_model(self):
        class LSTMDQN(nn.Module):
            def __init__(self, state_size, action_size):
                super(LSTMDQN, self).__init__()
                # LSTM Layer (now 3 layers)
                self.lstm = nn.LSTM(
                    input_size=state_size,
                    hidden_size=256,  # Increased hidden size
                    num_layers=1,  # Changed from 2 to 3 layers
                    batch_first=True,
                    dropout=0.0  # Dropout applies between LSTM layers except the last
                )
                # Fully Connected Layers
                self.fc1 = nn.Linear(256, 128)  # Hidden Layer 2
                self.fc2 = nn.Linear(128, 64)  # Hidden Layer 3
                self.fc3 = nn.Linear(64, action_size)  # Output Layer

            def forward(self, x):
                # Reshape input for LSTM
                if len(x.shape) == 1:
                    x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, state_size)
                elif len(x.shape) == 2:
                    x = x.unsqueeze(1)  # (batch_size, 1, state_size)

                # LSTM Forward Pass (3 layers)
                h0 = torch.zeros(1, x.size(0), 256).to(device)  # 3 layers
                c0 = torch.zeros(1, x.size(0), 256).to(device)
                out, _ = self.lstm(x, (h0, c0))
                out = out[:, -1, :]  # Take last output

                # Fully Connected Layers
                out = torch.relu(self.fc1(out))
                out = torch.relu(self.fc2(out))
                return self.fc3(out)  # Raw Q-values

        return LSTMDQN(self.state_size, self.action_size)

    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Convert state to tensor and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0

        minibatch = random.sample(self.memory, batch_size)

        # Prepare batch data
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).to(device)

        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss
        criterion = nn.MSELoss()
        loss = criterion(current_q.squeeze(), target_q)

        # Optimize the model
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()

        # Optional: Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
           #self.epsilon = max(self.epsilon_min, self.epsilon - 1e-3)

        return loss.item()


class MultiAgentController:
    def __init__(self, env):
        self.env = env
        # Initialize all three agents with proper action sizes
        self.agent1 = DQNAgent(state_size=14,
                               action_size=len(env.ActionInfo[0].values),
                               agent_id=1)
        self.agent2 = DQNAgent(state_size=14,
                               action_size=len(env.ActionInfo[1].values),
                               agent_id=2)
        self.batch_size = 128  # Increased for better GPU utilization

    def get_action(self, state):
        # Each agent processes the same state but selects its own action
        action1_idx = self.agent1.act(state[0])
        action2_idx = self.agent2.act(state[1])

        # Convert indices to actual action values
        action1 = self.env.ActionInfo[0].values[action1_idx]
        action2 = self.env.ActionInfo[1].values[action2_idx]

        return [action1, action2], [action1_idx, action2_idx]

    def remember(self, state, action_indices, rewards, next_state, done):
        # Each agent remembers its own experience
        self.agent1.remember(state[0], action_indices[0], rewards[0], next_state[0], done)
        self.agent2.remember(state[1], action_indices[1], rewards[1], next_state[1], done)

    def replay(self):
        # Each agent learns from its own experience
        loss1 = self.agent1.replay(self.batch_size)
        loss2 = self.agent2.replay(self.batch_size)
        return (loss1 or 0, loss2 or 0)  # Handle None returns

    def update_target_models(self):
        self.agent1.update_target_model()
        self.agent2.update_target_model()


class TrainingLogger:
    def __init__(self):
        self.history = {
            'episode': [],
            'rewards': {'agent1': [], 'agent2': []},
            'losses': {'agent1': [], 'agent2': []},
            'success': [],
            'energy': []
        }
        self.start_time = time()

    def log_episode(self, episode, rewards, losses, env):
        self.history['episode'].append(episode)
        for i, agent in enumerate(['agent1', 'agent2']):
            self.history['rewards'][agent].append(rewards[i])
            self.history['losses'][agent].append(losses[i] if losses[i] is not None else 0)

        # Calculate metrics
        success_rate = np.sum(env.success[-100:]) if len(env.success) > 0 else 0
        energy = np.sum(env.energy_consumption[-100:]) if len(env.energy_consumption) > 0 else 0
        self.history['success'].append(success_rate)
        self.history['energy'].append(energy)

    def plot_progress(self, live_update=False):
        plt.figure(figsize=(15, 10)) if not live_update else None

        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.cla()
        for agent in self.history['rewards']:
            plt.plot(self.history['episode'], self.history['rewards'][agent], label=agent)
        plt.title('Agent Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)

        # Plot losses
        plt.subplot(2, 2, 2)
        plt.cla()
        for agent in self.history['losses']:
            plt.plot(self.history['episode'], self.history['losses'][agent], label=agent)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Success rate
        plt.subplot(2, 2, 3)
        plt.cla()
        window = max(1, len(self.history['success']) // 20)
        smoothed = np.convolve(self.history['success'], np.ones(window) / window, mode='valid')
        plt.plot(self.history['episode'][:len(smoothed)], smoothed)
        plt.title(f'Success number (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Rate')
        plt.grid(True)

        # Energy efficiency
        plt.subplot(2, 2, 4)
        plt.cla()
        plt.plot(self.history['episode'], self.history['energy'])
        plt.title('Energy')
        plt.xlabel('Episode')
        plt.ylabel('Energy consumption')
        plt.grid(True)

        plt.tight_layout()
        if live_update:
            plt.pause(0.01)
            plt.draw()
        else:
            plt.savefig(f'training_results_{int(time())}.png', dpi=300)
            plt.close()

    def save_history(self):
        import pandas as pd
        df = pd.DataFrame({
            'episode': self.history['episode'],
            'agent1_reward': self.history['rewards']['agent1'],
            'agent2_reward': self.history['rewards']['agent2'],
            'success_rate': self.history['success'],
            'energy': self.history['energy']
        })
        df.to_csv('training_history.csv', index=False)


# [Keep your existing DQNAgent, MultiAgentController, and gap_region_dqn classes here]
# [Make sure they're exactly as shown in previous working code]

def train(env, controller, episodes=200):
    logger = TrainingLogger()

    # Define signal handler
    def signal_handler(sig, frame):
        print("\nTraining interrupted! Saving models and plots...")
        # Final save
        logger.plot_progress(live_update=False)
        logger.save_history()
        # Save models
        torch.save(controller.agent1.model.state_dict(), 'agent1_model_interrupted.pth')
        torch.save(controller.agent2.model.state_dict(), 'agent2_model_interrupted.pth')
        print("Models and plots saved. Exiting...")
        sys.exit(0)

    # Set the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        for e in tqdm(range(episodes), desc="Training"):
            state, _ = env.reset()
            episode_rewards = [0, 0, 0]
            done = False

            while not done:
                action, action_indices = controller.get_action(state)
                next_state, rewards, done = env.step(action)
                controller.remember(state, action_indices, rewards, next_state, done)
                losses = controller.replay()

                episode_rewards = [sum(x) for x in zip(episode_rewards, rewards)]
                state = next_state

            # Logging and visualization
            logger.log_episode(e, episode_rewards, losses, env)

            if e % 10 == 0:  # Update plots every 10 episodes
                logger.plot_progress(live_update=True)

            if e % 100 == 0:  # Save intermediate results
                logger.save_history()
                # Save intermediate models

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Final save
        logger.plot_progress(live_update=False)
        logger.save_history()
        # Save models
        torch.save(controller.agent1.model.state_dict(), 'agent1_model_final.pth')
        torch.save(controller.agent2.model.state_dict(), 'agent2_model_final.pth')
        print("Training complete. Models and plots saved.")

    return logger.history

if __name__ == "__main__":
    env = env_2agent()
    controller = MultiAgentController(env)

    print("Starting training with live monitoring...")
    history = train(env, controller, episodes=200)

    # Save models
    torch.save(controller.agent1.model.state_dict(), 'agent1_model.pth')
    torch.save(controller.agent2.model.state_dict(), 'agent2_model.pth')
    print("Training complete. Models and plots saved.")
