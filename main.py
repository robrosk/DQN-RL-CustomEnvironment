#!/usr/bin/env python3
"""
Frozen Lake Reinforcement Learning project with Deep Q-Network.
This script serves as the entry point to the project.
"""

import sys

import argparse
import os

from train import test, train

def main():
    """Main entry point for the application."""
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
    
    # Visualization parameters
    parser.add_argument("--render", action="store_true", help="Render the environment during testing")
    
    args = parser.parse_args()
    
    if args.test:
        if not os.path.exists(args.model):
            print(f"Error: Model file '{args.model}' not found!")
            print("Please train the model first or specify a valid model file.")
            return 1
            
        print("Testing agent...")
        test(args)
    else:
        print("Training agent...")
        train(args)
    
    print("Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 