import numpy as np
import torch
from env_2agent_constant import env_2agent
from two_agent import DQNAgent


def simulate(env, controller, episodes=1):
    """Run simulation with trained agents and calculate total averages"""
    total_success = 0
    total_energy = 0
    total_energy_efficiency = 0
    total_success_probability = 0

    for e in range(episodes):
        state, _ = env.reset()
        done = False
        episode_success = 0
        episode_energy = 0
        episode_energy_efficiency = 0
        episode_success_probability = 0
        step_count = 0

        print(f"\n=== Episode {e + 1} ===")

        while not done:
            action, _ = controller.get_action(state)
            next_state, rewards, done = env.step(action)
            state = next_state
            step_count += 1

            # Sum success and energy for current step
            if len(env.success) > 0:
                episode_success += env.success[-1]
            if len(env.success) > 0:
                episode_success_probability += env.success_probability[-1]
            if len(env.energy_consumption) > 0:
                episode_energy += env.energy_consumption[-1]
            if len(env.energy_consumption) > 0:
                episode_energy_efficiency += env.energyData[-1]

            print(f"Step {step_count}: Success={env.success[-1] if len(env.success)>0 else 0}, Energy={env.energy_consumption[-1] if len(env.energy_consumption)>0 else 0}")

        # Calculate episode averages (sum of all steps divided by steps)
        avg_episode_success = episode_success / 100 if step_count > 0 else 0
        avg_episode_energy = episode_energy / 100 if step_count > 0 else 0
        avg_episode_energy_efficiency = episode_energy_efficiency / 100 if step_count > 0 else 0
        avg_episode_success_probability = episode_success_probability / 100 if step_count > 0 else 0

        # Accumulate for total average
        total_success += avg_episode_success
        total_energy += avg_episode_energy
        total_energy_efficiency += avg_episode_energy_efficiency
        total_success_probability += avg_episode_success_probability

        print(f"\nEpisode {e+1} Results:")
        print(f"  Avg Success: {avg_episode_success:.5f}")
        print(f"  Avg Energy: {avg_episode_energy:.5f}")
        print(f"  Avg Energy_efficiency: {avg_episode_energy_efficiency:.5f}")
        print(f"  Avg Success_probability: {avg_episode_success_probability:.5f}")

    # Calculate final averages across all episodes
    final_avg_success = total_success / episodes
    final_avg_energy = total_energy / episodes
    final_avg_energy_efficiency = total_energy_efficiency / episodes
    final_avg_success_probability = total_success_probability / episodes

    print("\n=== FINAL RESULTS ===")
    print(f"Average Success across {episodes} episodes: {final_avg_success:.5f}")
    print(f"Average Energy across {episodes} episodes: {final_avg_energy:.5f}")
    print(f"Average Energy_efficiency across {episodes} episodes: {final_avg_energy_efficiency:.5f}")
    print(f"Average Success_probability across {episodes} episodes: {final_avg_success_probability:.5f}")

    return final_avg_success, final_avg_energy, final_avg_energy_efficiency, final_avg_success_probability



class SimMultiAgentController:
    """Modified controller for simulation (no training)"""

    def __init__(self, env, agent1_path, agent2_path):
        self.env = env

        # Initialize agents
        self.agent1 = DQNAgent(state_size=14,
                               action_size=len(env.ActionInfo[0].values),
                               agent_id=1)
        self.agent2 = DQNAgent(state_size=14,
                               action_size=len(env.ActionInfo[1].values),
                               agent_id=2)

        # Load trained models
        self.agent1.model.load_state_dict(torch.load(agent1_path))
        self.agent2.model.load_state_dict(torch.load(agent2_path))

        # Set epsilon to 0 (no exploration)
        self.agent1.epsilon = 0
        self.agent2.epsilon = 0

        # Put models in evaluation mode
        self.agent1.model.eval()
        self.agent2.model.eval()

    def get_action(self, state):
        # Each agent processes the same state but selects its own action
        action1_idx = self.agent1.act(state[0])
        action2_idx = self.agent2.act(state[1])

        # Convert indices to actual action values
        action1 = self.env.ActionInfo[0].values[action1_idx]
        action2 = self.env.ActionInfo[1].values[action2_idx]

        return [action1, action2], [action1_idx, action2_idx]


if __name__ == "__main__":
    # Initialize environment
    env = env_2agent()

    # Paths to your saved models
    agent1_path = 'agent1_model_final.pth'  # Update with your actual path
    agent2_path = 'agent2_model_final.pth'  # Update with your actual path

    # Create simulation controller
    controller = SimMultiAgentController(env, agent1_path, agent2_path)

    # Run simulation
    print("Starting simulation with trained agents...")
    simulate(env, controller, episodes=1)
