import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

from replay_buffer import ReplayBuffer

class DQNAgent:
    """Deep Q-Network Agent."""
    
    def __init__(self, state_size, action_size, 
                 learning_rate=0.001, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=64, update_target_every=100):
        """Initialize a DQN Agent object.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Number of possible actions
            learning_rate (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Exploration rate
            epsilon_decay (float): Rate at which epsilon decays
            epsilon_min (float): Minimum exploration rate
            batch_size (int): Size of each training batch
            update_target_every (int): How often to update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.learning_rate = learning_rate
        
        # Counter to track when to update target network
        self.update_counter = 0
        
        # Create replay memory
        self.memory = ReplayBuffer(batch_size=batch_size)
        
        # Build networks
        self.q_network = self._build_network()
        self.target_q_network = self._build_network()
        
        # Initialize target network weights to match q_network
        self.update_target_network()
        
    def _build_network(self):
        """Build a neural network model for approximating the Q-function.
        
        Returns:
            tf.keras.Sequential: A neural network model
        """
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(128, activation='relu'),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])

        model.compile(
            loss=Huber(),
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        )
        return model
    
    def update_target_network(self):
        """Update the target network with the weights from the primary network."""
        self.target_q_network.set_weights(self.q_network.get_weights())
    
    def get_action(self, state):
        """Return an action using an epsilon-greedy policy.
        
        Args:
            state: The current state
            
        Returns:
            int: The action to take
        """
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])
    
    def add_experience(self, state, action, reward, next_state, done):
        """Add an experience to memory.
        
        Args:
            state: Current state
            action (int): Action taken
            reward (float): Reward received
            next_state: State after taking action
            done (bool): Whether the episode is done
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def train(self):
        """Update the Q-Network using a batch of experience from memory."""
        # If memory doesn't have enough experience yet, don't train
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences from memory
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Calculate current Q values from primary network
        current_q = self.q_network.predict(states, verbose=0)
        
        # Calculate target Q values from target network
        target_q = np.copy(current_q)

        # Get next Q values from target and primary network for Double DQN
        next_q_target = self.target_q_network.predict(next_states, verbose=0)
        next_actions = np.argmax(self.q_network.predict(next_states, verbose=0), axis=1)

        # Update target Q values based on Double DQN Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                best_next = next_q_target[i, next_actions[i]]
                target_q[i, actions[i]] = rewards[i] + self.gamma * best_next
        
        # Train the network
        self.q_network.fit(states, target_q, epochs=1, verbose=0)
        
        # Update target network counter
        self.update_counter += 1
        
        # Update target network if needed
        if self.update_counter % self.update_target_every == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        """Save the Q-Network model to a file.
        
        Args:
            filename (str): File path to save the model
        """
        try:
            self.q_network.save(filename)
        except Exception as e:
            print(f"Full model saving failed: {e}. Saving weights only...")
            
            weight_filename = filename.replace('.h5', '_weights.h5')
            self.q_network.save_weights(weight_filename)
            print(f"Weights saved to {weight_filename}")
            
            config_filename = filename.replace('.h5', '_config.json')
            with open(config_filename, 'w') as f:
                f.write(self.q_network.to_json())
            print(f"Model configuration saved to {config_filename}")
    
    def load(self, filename):
        """Load the Q-Network model from a file.
        
        Args:
            filename (str): File path to load the model from
        """
        try:
            # Try loading the model directly
            self.q_network = tf.keras.models.load_model(filename)
        except (TypeError, ValueError) as e:
            print(f"Standard loading failed, trying alternative method: {e}")
            try:
                temp_model = self._build_network()
                
                temp_model.load_weights(filename)
                self.q_network = temp_model
            except:
                print("Trying to load weights only...")
                self.q_network = self._build_network()
                self.q_network.load_weights(filename)
        
        self.q_network.compile(
            loss=Huber(),
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        )
        
        self.update_target_network() 