"""
Deep Q-Learning (DQN) Implementation for Battery-Optimized Cryptographic Selection
=================================================================================

This module implements Deep Q-Network with experience replay, target networks,
and advanced training techniques for the cryptographic algorithm selection problem.

Key Features:
- Neural network Q-function approximation
- Experience replay buffer for stable learning
- Target network for improved convergence
- Warm-start initialization from expert knowledge
- Advanced visualization and analysis

Author: RL Team
Date: September 4, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from environment.state_space import StateSpace, CryptoState, CryptoAlgorithm
from environment.crypto_environment import CryptoEnvironment

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    """
    Deep Q-Network Architecture
    
    Input: State representation (10 dimensions - one-hot encoded)
    Output: Q-values for 8 actions (algorithms)
    """
    
    def __init__(self, input_size: int = 10, hidden_sizes: List[int] = [128, 64, 32], output_size: int = 8):
        """
        Initialize DQN architecture
        
        Args:
            input_size: State representation size (10 for our one-hot encoding)
            hidden_sizes: List of hidden layer sizes
            output_size: Action space size (8 algorithms)
        """
        super(DQN, self).__init__()
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)  # Prevent overfitting
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through network"""
        return self.network(x)

class ExperienceReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int = 100000):
        """Initialize replay buffer"""
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Deep Q-Network Agent with Advanced Features
    
    Features:
    - Neural network Q-function approximation
    - Experience replay for stable learning
    - Target network for improved convergence
    - Warm-start from expert knowledge
    - Advanced training techniques
    """
    
    def __init__(
        self,
        state_space: StateSpace,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        buffer_capacity: int = 100000,
        target_update_frequency: int = 100,
        hidden_sizes: List[int] = [128, 64, 32],
        warm_start: bool = True,
        device: str = None
    ):
        """
        Initialize DQN agent
        
        Args:
            state_space: State space manager
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate
            batch_size: Batch size for training
            buffer_capacity: Experience replay buffer size
            target_update_frequency: How often to update target network
            hidden_sizes: Neural network architecture
            warm_start: Whether to initialize with expert knowledge
            device: Device to run on (cuda/cpu)
        """
        
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize networks
        self.q_network = DQN(input_size=10, hidden_sizes=hidden_sizes, output_size=8).to(self.device)
        self.target_network = DQN(input_size=10, hidden_sizes=hidden_sizes, output_size=8).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.training_losses = []
        self.training_rewards = []
        self.expert_agreement_history = []
        self.q_value_history = []
        self.episode_lengths = []
        self.steps_done = 0
        
        # Initialize with expert knowledge if requested
        if warm_start:
            self._initialize_warm_start()
        
        print(f"🧠 DQN Agent initialized")
        print(f"   Network Architecture: 10 → {' → '.join(map(str, hidden_sizes))} → 8")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Buffer Capacity: {buffer_capacity:,}")
        print(f"   Target Update Frequency: {target_update_frequency}")
        print(f"   Warm Start: {'✅' if warm_start else '❌'}")
    
    def _initialize_warm_start(self):
        """Initialize network with expert knowledge using supervised learning"""
        print("🎯 Initializing DQN with expert knowledge...")
        
        # Collect expert demonstrations
        expert_states = []
        expert_actions = []
        
        for state_idx in range(30):
            state = CryptoState.from_index(state_idx)
            expert_action = self.state_space.get_expert_action(state)
            
            # Convert to network input format
            state_obs = self._state_to_observation(state)
            expert_states.append(state_obs)
            expert_actions.append(expert_action.value)
        
        # Convert to tensors (efficient conversion)
        states_array = np.array(expert_states)
        actions_array = np.array(expert_actions)
        states_tensor = torch.FloatTensor(states_array).to(self.device)
        actions_tensor = torch.LongTensor(actions_array).to(self.device)
        
        # Pre-training with expert data
        criterion = nn.CrossEntropyLoss()
        pre_train_optimizer = optim.Adam(self.q_network.parameters(), lr=0.01)
        
        print("   Pre-training on expert demonstrations...")
        for epoch in range(100):
            # Forward pass
            q_values = self.q_network(states_tensor)
            loss = criterion(q_values, actions_tensor)
            
            # Backward pass
            pre_train_optimizer.zero_grad()
            loss.backward()
            pre_train_optimizer.step()
            
            if (epoch + 1) % 25 == 0:
                accuracy = self._calculate_expert_accuracy(states_tensor, actions_tensor)
                print(f"   Epoch {epoch + 1}: Loss = {loss.item():.4f}, Accuracy = {accuracy:.1%}")
        
        # Update target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        final_accuracy = self._calculate_expert_accuracy(states_tensor, actions_tensor)
        print(f"✅ Warm-start completed. Final expert accuracy: {final_accuracy:.1%}")
    
    def _calculate_expert_accuracy(self, states_tensor, actions_tensor):
        """Calculate accuracy against expert actions"""
        with torch.no_grad():
            q_values = self.q_network(states_tensor)
            predicted_actions = q_values.argmax(dim=1)
            correct = (predicted_actions == actions_tensor).float()
            return correct.mean().item()
    
    def _state_to_observation(self, state: CryptoState) -> np.ndarray:
        """Convert state to observation vector (same as environment)"""
        obs = np.zeros(10)
        obs[state.battery_level.value] = 1.0
        obs[5 + state.threat_status.value] = 1.0
        obs[8 + state.mission_criticality.value] = 1.0
        return obs
    
    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with neural network
        
        Args:
            observation: Current state observation
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index (0-7)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, 7)
        else:
            # Exploit: best action from neural network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step using experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors (efficient conversion)
        states_array = np.array([e.state for e in batch])
        actions_array = np.array([e.action for e in batch])
        rewards_array = np.array([e.reward for e in batch])
        next_states_array = np.array([e.next_state for e in batch])
        dones_array = np.array([e.done for e in batch])
        
        states = torch.FloatTensor(states_array).to(self.device)
        actions = torch.LongTensor(actions_array).to(self.device)
        rewards = torch.FloatTensor(rewards_array).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        dones = torch.BoolTensor(dones_array).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps_done % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.steps_done += 1
        self.training_losses.append(loss.item())
        
        return loss.item()
    
    def train(
        self,
        environment: CryptoEnvironment,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 100,
        evaluation_frequency: int = 100
    ):
        """
        Train the DQN agent
        
        Args:
            environment: Training environment
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            evaluation_frequency: How often to evaluate performance
        """
        print(f"🚀 Starting DQN training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Reset environment
            obs, info = environment.reset()
            state_index = environment.get_state_index()
            
            episode_reward = 0
            episode_steps = 0
            expert_agreements = 0
            episode_q_values = []
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.select_action(obs, training=True)
                
                # Take step
                next_obs, reward, terminated, truncated, info = environment.step(action)
                done = terminated or truncated
                
                # Store experience
                self.store_experience(obs, action, reward, next_obs, done)
                
                # Train network
                loss = self.train_step()
                
                # Track metrics
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    q_values = self.q_network(obs_tensor)
                    episode_q_values.append(q_values.max().item())
                
                # Track expert agreement
                expert_action = info['expert_action']
                if CryptoAlgorithm(action).name == expert_action:
                    expert_agreements += 1
                
                episode_reward += reward
                episode_steps += 1
                obs = next_obs
                
                if done:
                    break
            
            # Decay exploration
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Track episode metrics
            self.training_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.expert_agreement_history.append(expert_agreements / episode_steps if episode_steps > 0 else 0)
            if episode_q_values:
                self.q_value_history.append(np.mean(episode_q_values))
            
            # Print progress
            if (episode + 1) % evaluation_frequency == 0:
                avg_reward = np.mean(self.training_rewards[-evaluation_frequency:])
                avg_agreement = np.mean(self.expert_agreement_history[-evaluation_frequency:])
                avg_loss = np.mean(self.training_losses[-1000:]) if self.training_losses else 0
                avg_q_value = np.mean(self.q_value_history[-evaluation_frequency:]) if self.q_value_history else 0
                
                print(f"Episode {episode + 1:4d}: "
                      f"Avg Reward = {avg_reward:6.2f}, "
                      f"Expert Agreement = {avg_agreement:6.1%}, "
                      f"ε = {self.epsilon:.3f}, "
                      f"Avg Loss = {avg_loss:.4f}, "
                      f"Avg Q = {avg_q_value:.2f}")
        
        print("✅ DQN training completed!")
        self._print_training_summary()
    
    def _print_training_summary(self):
        """Print training summary statistics"""
        print("\n" + "="*60)
        print("📊 DQN TRAINING SUMMARY")
        print("="*60)
        
        if self.training_rewards:
            print(f"📈 Final Avg Reward (last 100): {np.mean(self.training_rewards[-100:]):.3f}")
            print(f"🏆 Best Episode Reward: {max(self.training_rewards):.3f}")
            print(f"📉 Worst Episode Reward: {min(self.training_rewards):.3f}")
        
        if self.expert_agreement_history:
            final_agreement = np.mean(self.expert_agreement_history[-100:])
            print(f"🎯 Final Expert Agreement: {final_agreement:.1%}")
        
        if self.training_losses:
            print(f"📊 Final Loss: {np.mean(self.training_losses[-100:]):.6f}")
        
        if self.q_value_history:
            print(f"🔢 Final Avg Q-Value: {np.mean(self.q_value_history[-100:]):.3f}")
        
        print(f"🔍 Final Exploration Rate: {self.epsilon:.4f}")
        print(f"💾 Experiences Collected: {len(self.replay_buffer):,}")
        print(f"🏃 Training Steps: {self.steps_done:,}")
        
        print("="*60)
    
    def evaluate(self, environment: CryptoEnvironment, num_episodes: int = 100) -> Dict:
        """
        Evaluate trained DQN agent performance
        
        Args:
            environment: Test environment
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"🧪 Evaluating DQN agent over {num_episodes} episodes...")
        
        self.q_network.eval()  # Set to evaluation mode
        
        eval_rewards = []
        eval_expert_agreements = []
        eval_episode_lengths = []
        action_distribution = {}
        q_value_analysis = []
        
        for episode in range(num_episodes):
            obs, info = environment.reset()
            
            episode_reward = 0
            expert_agreements = 0
            steps = 0
            episode_q_values = []
            
            while steps < 100:  # Max steps
                action = self.select_action(obs, training=False)  # No exploration
                obs, reward, terminated, truncated, info = environment.step(action)
                
                # Track Q-values
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    q_values = self.q_network(obs_tensor)
                    episode_q_values.extend(q_values.squeeze().cpu().numpy())
                
                episode_reward += reward
                steps += 1
                
                # Track metrics
                algorithm_name = CryptoAlgorithm(action).name
                action_distribution[algorithm_name] = action_distribution.get(algorithm_name, 0) + 1
                
                expert_action = info['expert_action']
                if algorithm_name == expert_action:
                    expert_agreements += 1
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_expert_agreements.append(expert_agreements / steps if steps > 0 else 0)
            eval_episode_lengths.append(steps)
            q_value_analysis.extend(episode_q_values)
        
        self.q_network.train()  # Back to training mode
        
        # Calculate metrics
        results = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_expert_agreement': np.mean(eval_expert_agreements),
            'avg_episode_length': np.mean(eval_episode_lengths),
            'action_distribution': action_distribution,
            'q_value_stats': {
                'mean': np.mean(q_value_analysis),
                'std': np.std(q_value_analysis),
                'min': np.min(q_value_analysis),
                'max': np.max(q_value_analysis)
            }
        }
        
        self._print_evaluation_results(results)
        return results
    
    def _print_evaluation_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("🏆 DQN EVALUATION RESULTS")
        print("="*60)
        
        print(f"📊 Average Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"🎯 Expert Agreement: {results['avg_expert_agreement']:.1%}")
        print(f"⏱️  Average Episode Length: {results['avg_episode_length']:.1f}")
        
        print(f"\n🔢 Q-Value Statistics:")
        q_stats = results['q_value_stats']
        print(f"   Mean: {q_stats['mean']:.3f} ± {q_stats['std']:.3f}")
        print(f"   Range: [{q_stats['min']:.3f}, {q_stats['max']:.3f}]")
        
        print("\n🎮 Action Distribution:")
        total_actions = sum(results['action_distribution'].values())
        for action, count in sorted(results['action_distribution'].items()):
            percentage = count / total_actions * 100
            print(f"   {action:>12}: {count:4d} ({percentage:5.1f}%)")
        
        print("="*60)
    
    def save_model(self, filepath: str):
        """Save trained DQN model"""
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses,
            'expert_agreement_history': self.expert_agreement_history,
            'q_value_history': self.q_value_history,
            'parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_frequency': self.target_update_frequency
            },
            'steps_done': self.steps_done
        }
        
        torch.save(model_data, filepath)
        print(f"💾 DQN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained DQN model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(model_data['q_network_state_dict'])
        self.target_network.load_state_dict(model_data['target_network_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
        self.training_rewards = model_data['training_rewards']
        self.training_losses = model_data['training_losses']
        self.expert_agreement_history = model_data['expert_agreement_history']
        self.q_value_history = model_data['q_value_history']
        self.steps_done = model_data['steps_done']
        
        # Load parameters
        params = model_data['parameters']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']
        self.epsilon = params['epsilon']
        self.epsilon_end = params['epsilon_end']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.target_update_frequency = params['target_update_frequency']
        
        print(f"📂 DQN model loaded from {filepath}")

if __name__ == "__main__":
    # Test DQN implementation
    print("🚀 Testing DQN Implementation")
    
    # Initialize components
    state_space = StateSpace()
    environment = CryptoEnvironment(random_seed=42)
    
    # Create DQN agent
    dqn_agent = DQNAgent(
        state_space=state_space,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon_start=0.9,
        epsilon_end=0.01,
        batch_size=32,
        hidden_sizes=[128, 64, 32],
        warm_start=True
    )
    
    # Quick training test
    print("\n🏃‍♂️ Quick DQN training test (100 episodes)...")
    dqn_agent.train(environment, num_episodes=100, evaluation_frequency=50)
    
    # Evaluate
    results = dqn_agent.evaluate(environment, num_episodes=20)
    
    print("\n✅ DQN test completed!")
