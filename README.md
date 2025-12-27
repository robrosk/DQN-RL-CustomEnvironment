# Frozen Lake Reinforcement Learning

This project implements a Deep Q-Network (DQN) agent to solve the Frozen Lake environment. The environment now supports procedurally generated grids so the agent can generalize across many different lakes instead of memorizing a single layout.

The original implementation was developed in Jupyter notebooks. The code has since been refactored into Python modules with proper classes for improved modularity and reusability.

## Project Structure

- `environment.py`: Contains the Environment class for solvable, randomly generated Frozen Lake maps
- `replay_buffer.py`: Implements the ReplayBuffer class for experience replay
- `agent.py`: Defines the DQNAgent class with the neural network for Q-learning
- `train.py`: Contains training and testing logic
- `main.py`: Entry point for the application

## Features

- Deep Q-Network (DQN) implementation with TensorFlow
- Custom environment built from scratch with on-reset map randomization
- Experience replay for efficient learning
- Target network for stable training
 - Optional pygame visualization for a game-like view

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/frozen-lake-rl.git
cd frozen-lake-rl
```

2. Install the required packages in a virtual environment:
```
# Create and activate a virtual environment
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

To train a new agent:

```
python main.py --episodes 1000 --max-steps 100
```

Training parameters:
- `--episodes`: Number of episodes to train (default: 1000)
- `--max-steps`: Maximum steps per episode (default: 100)
- `--learning-rate`: Learning rate for the agent (default: 0.001)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon-decay`: Exploration rate decay (default: 0.995)
- `--epsilon-min`: Minimum exploration rate (default: 0.01)
- `--batch-size`: Batch size for training (default: 64)
- `--update-target-every`: Update target network every N steps (default: 100)

Environment parameters:
- `--grid-size`: Size of the square grid (default: 4)
- `--hole-prob`: Probability of placing a hole in each non-terminal cell (default: 0.2)
- `--slip-prob`: Probability of slipping to a random move (default: 0.2)
- `--min-path-ratio`: Enforce a minimum shortest-path length relative to grid size to keep maps challenging (default: 1.25)
- `--static-map`: Use a single generated map for all episodes (by default a fresh map is generated each reset)
- `--seed`: Optional random seed for reproducibility

The trained model will be saved to `models/dqn_final.h5`.

### Testing

To test a trained agent:

```
python main.py --test --model models/dqn_final.h5
```

Testing parameters:
- `--test`: Flag to indicate testing mode
- `--model`: Path to the trained model (default: models/dqn_final.h5)
- `--test-episodes`: Number of test episodes (default: 5)
- `--render`: Flag to render the environment during testing

## Environment

The Frozen Lake environment is a 4x4 grid:
- S: Start position (0,0)
- F: Frozen surface (safe)
- H: Hole (terminates episode with negative reward)
- G: Goal (terminates episode with positive reward)

It's a stochastic environment, where the agent has a configurable chance of slipping to a random direction regardless of the chosen action. The map is regenerated to remain solvable while meeting a minimum difficulty threshold.

## Agent

The agent uses a Deep Q-Network with the following features:
- Experience replay for more efficient learning
- Target network to stabilize training
- Epsilon-greedy exploration strategy

## Troubleshooting

If you encounter issues with TensorFlow installation:
1. Make sure you're using a compatible Python version (3.6-3.9 work best with TensorFlow)
2. You can try installing a specific version: `pip install tensorflow==2.8.0`
3. If you're on Windows, ensure you have the Microsoft Visual C++ Redistributable installed
4. Consider using a CPU-only version if you don't have a compatible GPU: `pip install tensorflow-cpu`

