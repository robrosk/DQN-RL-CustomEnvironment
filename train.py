import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from agent import DQNAgent
from environment import Environment

def plot_training_rewards(rewards, avg_window=100, filename=None):
    """Plot the rewards per episode during training.
    
    Args:
        rewards (list): List of rewards per episode
        avg_window (int): Window size for the moving average
        filename (str, optional): If provided, save the plot to this file
    """
    # Compute moving average
    moving_avg = [np.mean(rewards[max(0, i-avg_window):i+1]) for i in range(len(rewards))]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Episode Reward")
    plt.plot(moving_avg, label=f"{avg_window}-Episode Moving Average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.grid(True)
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def train(args):
    """Train a DQN agent on the Frozen Lake environment.
    
    Args:
        args: Command line arguments
    """
    # Create the environment
    env = Environment(
        grid_size=args.grid_size,
        hole_prob=args.hole_prob,
        slip_prob=args.slip_prob,
        randomize_on_reset=not args.static_map,
        min_path_length_ratio=args.min_path_ratio,
        seed=args.seed,
    )
    
    # Create the agent
    agent = DQNAgent(
        state_size=env.obs_size,
        action_size=env.n_actions,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        update_target_every=args.update_target_every
    )
    
    # Training setup
    n_episodes = args.episodes
    max_steps = args.max_steps
    rewards_history = []
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Training loop
    for episode in tqdm(range(n_episodes), desc="Training"):
        # Reset environment
        state_tuple = env.reset()
        state = env.state_to_one_hot(state_tuple)
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step in environment
            actual_action, next_state_tuple, reward, done = env.step(action)
            next_state = env.state_to_one_hot(next_state_tuple)
            
            # Store experience in replay memory
            agent.add_experience(state, actual_action, reward, next_state, done)
            
            # Train the agent
            agent.train()
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # If episode is done, break
            if done:
                break
        
        # Save episode reward
        rewards_history.append(total_reward)
        
        # Occasionally save the model
        if (episode + 1) % 100 == 0:
            agent.save(f"models/dqn_episode_{episode+1}.h5")
    
    # Save final model
    agent.save("models/dqn_final.h5")
    
    # Plot rewards
    plot_training_rewards(rewards_history, avg_window=100, filename="training_rewards.png")
    
    # Save rewards history
    np.save("rewards_history.npy", np.array(rewards_history))
    
    return agent, rewards_history

def test(args):
    """Test a trained DQN agent on the Frozen Lake environment.

    Args:
        args: Command line arguments
    """
    # Create the environment
    env = Environment(
        grid_size=args.grid_size,
        hole_prob=args.hole_prob,
        slip_prob=args.slip_prob,
        randomize_on_reset=not args.static_map,
        min_path_length_ratio=args.min_path_ratio,
        seed=args.seed,
    )
    
    # Create the agent
    agent = DQNAgent(
        state_size=env.obs_size,
        action_size=env.n_actions
    )
    
    # Load the trained model
    agent.load(args.model)
    
    # Testing setup
    n_episodes = args.test_episodes
    max_steps = args.max_steps
    rewards_history = []
    
    # Testing loop
    for episode in range(n_episodes):
        # Reset environment
        state_tuple = env.reset()
        state = env.state_to_one_hot(state_tuple)
        total_reward = 0
        
        print(f"Episode {episode+1}/{n_episodes}")
        print(env.render_text())
        
        for step in range(max_steps):
            # Get action from agent (no exploration during testing)
            agent.epsilon = 0.0
            action = agent.get_action(state)
            
            # Take step in environment
            actual_action, next_state_tuple, reward, done = env.step(action)
            next_state = env.state_to_one_hot(next_state_tuple)
            
            print(f"Action: {action}, State: {next_state_tuple}, Reward: {reward}")
            print(env.render_text())
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # If episode is done, break
            if done:
                break
        
        print(f"Episode {episode+1} finished with reward {total_reward}")
        print("-" * 40)
        
        # Save episode reward
        rewards_history.append(total_reward)
    
    # Print average reward
    print(f"Average reward over {n_episodes} episodes: {np.mean(rewards_history)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a DQN agent on Frozen Lake.")

    # Training parameters
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the agent")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Exploration rate decay")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum exploration rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--update-target-every", type=int, default=100, help="Update target network every N steps")

    # Environment parameters
    parser.add_argument("--grid-size", type=int, default=4, help="Size of the frozen lake grid")
    parser.add_argument("--hole-prob", type=float, default=0.2, help="Probability of a hole in each cell")
    parser.add_argument("--slip-prob", type=float, default=0.2, help="Probability of slipping to a random move")
    parser.add_argument(
        "--min-path-ratio",
        type=float,
        default=1.25,
        help="Shortest-path length must exceed grid_size * ratio to keep maps challenging",
    )
    parser.add_argument(
        "--static-map",
        action="store_true",
        help="Use a single generated map for all episodes instead of regenerating on reset",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    # Testing parameters
    parser.add_argument("--test", action="store_true", help="Test the agent instead of training")
    parser.add_argument("--model", type=str, default="models/dqn_final.h5", help="Model file to load for testing")
    parser.add_argument("--test-episodes", type=int, default=5, help="Number of test episodes")

    args = parser.parse_args()

    if args.test:
        test(args)
    else:
        train(args)