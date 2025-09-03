"""
Q-Learning Algorithm for Battery-Optimized Cryptographic Selection
=================================================================

This module implements the Q-Learning algorithm with warm-start initialization
from expert knowledge (lookup table).

Author: RL Team
Date: September 4, 2025
"""

import numpy as np
import json
import pickle
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from environment.state_space import StateSpace, CryptoState, CryptoAlgorithm
from environment.crypto_environment import CryptoEnvironment

class QLearningAgent:
    """
    Q-Learning Agent with Warm-Start Capability
    
    Features:
    - Warm-start initialization from expert knowledge
    - Epsilon-greedy exploration with decay
    - Experience replay for stable learning
    - Performance tracking and visualization
    """
    
    def __init__(
        self,
        state_space: StateSpace,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 0.1,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        warm_start: bool = True
    ):
        """
        Initialize Q-Learning agent
        
        Args:
            state_space: State space manager
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate  
            epsilon_decay: Exploration decay rate
            warm_start: Whether to initialize with expert knowledge
        """
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: Q(s, a) for all state-action pairs
        self.q_table = np.zeros((30, 8))  # 30 states × 8 actions
        
        # Performance tracking
        self.training_rewards = []
        self.training_losses = []
        self.expert_agreement_history = []
        self.episode_lengths = []
        
        # Experience buffer for analysis
        self.experience_buffer = deque(maxlen=10000)
        
        # Initialize with expert knowledge if requested
        if warm_start:
            self._initialize_warm_start()
        
        print(f"🧠 Q-Learning Agent initialized")
        print(f"   Learning Rate: {learning_rate}")
        print(f"   Discount Factor: {discount_factor}")
        print(f"   Exploration: {epsilon_start} → {epsilon_end}")
        print(f"   Warm Start: {'✅' if warm_start else '❌'}")
    
    def _initialize_warm_start(self):
        """Initialize Q-values using expert knowledge"""
        print("🎯 Initializing with expert knowledge...")
        
        expert_q_value = 10.0  # High value for expert choices
        alternative_q_value = 3.0  # Lower value for alternatives
        bad_choice_q_value = -5.0  # Negative for clearly bad choices
        
        for state_idx in range(30):
            state = CryptoState.from_index(state_idx)
            expert_action = self.state_space.get_expert_action(state)
            
            # Set expert choice to high value
            self.q_table[state_idx, expert_action.value] = expert_q_value
            
            # Set reasonable alternatives to medium value
            for action_idx in range(8):
                if action_idx != expert_action.value:
                    action = CryptoAlgorithm(action_idx)
                    
                    # Check if this is a reasonable alternative
                    if self._is_reasonable_alternative(state, action, expert_action):
                        self.q_table[state_idx, action_idx] = alternative_q_value
                    else:
                        self.q_table[state_idx, action_idx] = bad_choice_q_value
        
        print("✅ Warm-start initialization completed")
        self._print_warm_start_summary()
    
    def _is_reasonable_alternative(
        self, 
        state: CryptoState, 
        action: CryptoAlgorithm, 
        expert_action: CryptoAlgorithm
    ) -> bool:
        """Check if an action is a reasonable alternative to expert choice"""
        
        # Same type (pre/post quantum) is generally reasonable
        action_is_pqc = action.value >= 4
        expert_is_pqc = expert_action.value >= 4
        
        if action_is_pqc == expert_is_pqc:
            return True
        
        # Pre-quantum only acceptable with critical battery and no/low threat
        if (not action_is_pqc and 
            state.battery_level.value == 0 and 
            state.threat_status.value <= 1):
            return True
        
        # Post-quantum generally acceptable (safe choice)
        if action_is_pqc:
            return True
        
        return False
    
    def _print_warm_start_summary(self):
        """Print summary of warm-start initialization"""
        high_q_count = np.sum(self.q_table >= 8.0)
        medium_q_count = np.sum((self.q_table >= 2.0) & (self.q_table < 8.0))
        low_q_count = np.sum(self.q_table < 0)
        
        print(f"   High Q-values (≥8.0): {high_q_count} entries")
        print(f"   Medium Q-values (2.0-8.0): {medium_q_count} entries")
        print(f"   Low Q-values (<0): {low_q_count} entries")
    
    def select_action(self, state_index: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state_index: Current state index (0-29)
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action index (0-7)
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, 8)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state_index])
    
    def update_q_value(
        self, 
        state: int, 
        action: int, 
        reward: float, 
        next_state: int, 
        terminated: bool
    ):
        """
        Update Q-value using Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table[state, action]
        
        if terminated:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-learning update
        td_error = target_q - current_q
        new_q = current_q + self.learning_rate * td_error
        
        self.q_table[state, action] = new_q
        
        # Track training loss (TD error)
        self.training_losses.append(abs(td_error))
        
        return td_error
    
    def train(
        self, 
        environment: CryptoEnvironment, 
        num_episodes: int = 1000,
        max_steps_per_episode: int = 100,
        evaluation_frequency: int = 100
    ):
        """
        Train the Q-learning agent
        
        Args:
            environment: Training environment
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            evaluation_frequency: How often to evaluate performance
        """
        print(f"🚀 Starting Q-Learning training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Reset environment
            obs, info = environment.reset()
            state_index = environment.get_state_index()
            
            episode_reward = 0
            episode_steps = 0
            expert_agreements = 0
            
            for step in range(max_steps_per_episode):
                # Select action
                action = self.select_action(state_index, training=True)
                
                # Take step
                next_obs, reward, terminated, truncated, info = environment.step(action)
                next_state_index = environment.get_state_index()
                
                # Update Q-value
                td_error = self.update_q_value(
                    state_index, action, reward, next_state_index, terminated or truncated
                )
                
                # Track expert agreement
                expert_action = info['expert_action']
                if CryptoAlgorithm(action).name == expert_action:
                    expert_agreements += 1
                
                # Store experience
                self.experience_buffer.append({
                    'state': state_index,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state_index,
                    'terminated': terminated,
                    'td_error': td_error
                })
                
                episode_reward += reward
                episode_steps += 1
                state_index = next_state_index
                
                if terminated or truncated:
                    break
            
            # Decay exploration
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Track performance
            self.training_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.expert_agreement_history.append(expert_agreements / episode_steps)
            
            # Print progress
            if (episode + 1) % evaluation_frequency == 0:
                avg_reward = np.mean(self.training_rewards[-evaluation_frequency:])
                avg_agreement = np.mean(self.expert_agreement_history[-evaluation_frequency:])
                avg_loss = np.mean(self.training_losses[-1000:]) if self.training_losses else 0
                
                print(f"Episode {episode + 1:4d}: "
                      f"Avg Reward = {avg_reward:6.2f}, "
                      f"Expert Agreement = {avg_agreement:6.1%}, "
                      f"ε = {self.epsilon:.3f}, "
                      f"Avg TD Error = {avg_loss:.3f}")
        
        print("✅ Training completed!")
        self._print_training_summary()
    
    def _print_training_summary(self):
        """Print training summary statistics"""
        print("\n" + "="*60)
        print("📊 Q-LEARNING TRAINING SUMMARY")
        print("="*60)
        
        if self.training_rewards:
            print(f"📈 Final Avg Reward (last 100): {np.mean(self.training_rewards[-100:]):.3f}")
            print(f"🏆 Best Episode Reward: {max(self.training_rewards):.3f}")
            print(f"📉 Worst Episode Reward: {min(self.training_rewards):.3f}")
        
        if self.expert_agreement_history:
            final_agreement = np.mean(self.expert_agreement_history[-100:])
            print(f"🎯 Final Expert Agreement: {final_agreement:.1%}")
        
        if self.training_losses:
            print(f"📊 Final TD Error: {np.mean(self.training_losses[-100:]):.4f}")
        
        print(f"🔍 Final Exploration Rate: {self.epsilon:.4f}")
        print(f"💾 Q-table Non-zero Entries: {np.count_nonzero(self.q_table)}/240")
        
        print("="*60)
    
    def evaluate(self, environment: CryptoEnvironment, num_episodes: int = 100) -> Dict:
        """
        Evaluate trained agent performance
        
        Args:
            environment: Test environment
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"🧪 Evaluating agent over {num_episodes} episodes...")
        
        eval_rewards = []
        eval_expert_agreements = []
        eval_episode_lengths = []
        action_distribution = defaultdict(int)
        
        for episode in range(num_episodes):
            obs, info = environment.reset()
            state_index = environment.get_state_index()
            
            episode_reward = 0
            expert_agreements = 0
            steps = 0
            
            while steps < 100:  # Max steps
                action = self.select_action(state_index, training=False)  # No exploration
                obs, reward, terminated, truncated, info = environment.step(action)
                
                episode_reward += reward
                steps += 1
                
                # Track metrics
                action_distribution[CryptoAlgorithm(action).name] += 1
                expert_action = info['expert_action']
                if CryptoAlgorithm(action).name == expert_action:
                    expert_agreements += 1
                
                state_index = environment.get_state_index()
                
                if terminated or truncated:
                    break
            
            eval_rewards.append(episode_reward)
            eval_expert_agreements.append(expert_agreements / steps)
            eval_episode_lengths.append(steps)
        
        # Calculate metrics
        results = {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_expert_agreement': np.mean(eval_expert_agreements),
            'avg_episode_length': np.mean(eval_episode_lengths),
            'action_distribution': dict(action_distribution)
        }
        
        self._print_evaluation_results(results)
        return results
    
    def _print_evaluation_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "="*60)
        print("🏆 EVALUATION RESULTS")
        print("="*60)
        
        print(f"📊 Average Reward: {results['avg_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"🎯 Expert Agreement: {results['avg_expert_agreement']:.1%}")
        print(f"⏱️  Average Episode Length: {results['avg_episode_length']:.1f}")
        
        print("\n🎮 Action Distribution:")
        total_actions = sum(results['action_distribution'].values())
        for action, count in sorted(results['action_distribution'].items()):
            percentage = count / total_actions * 100
            print(f"   {action:>12}: {count:4d} ({percentage:5.1f}%)")
        
        print("="*60)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'q_table': self.q_table,
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses,
            'expert_agreement_history': self.expert_agreement_history,
            'parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.training_rewards = model_data['training_rewards']
        self.training_losses = model_data['training_losses']
        self.expert_agreement_history = model_data['expert_agreement_history']
        
        # Load parameters
        params = model_data['parameters']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']
        self.epsilon = params['epsilon']
        self.epsilon_end = params['epsilon_end']
        self.epsilon_decay = params['epsilon_decay']
        
        print(f"📂 Model loaded from {filepath}")
    
    def get_policy_summary(self) -> str:
        """Get summary of learned policy"""
        summary = []
        summary.append("🧠 LEARNED POLICY SUMMARY")
        summary.append("=" * 50)
        
        for state_idx in range(30):
            state = CryptoState.from_index(state_idx)
            best_action_idx = np.argmax(self.q_table[state_idx])
            best_action = CryptoAlgorithm(best_action_idx)
            best_q_value = self.q_table[state_idx, best_action_idx]
            
            expert_action = self.state_space.get_expert_action(state)
            match = "✅" if best_action == expert_action else "❌"
            
            summary.append(f"State {state_idx:2d}: {state.get_description()}")
            summary.append(f"   Learned: {best_action.name} (Q={best_q_value:.2f}) {match}")
            summary.append(f"   Expert:  {expert_action.name}")
            summary.append("")
        
        return "\n".join(summary)

if __name__ == "__main__":
    # Test Q-Learning implementation
    print("🚀 Testing Q-Learning Agent")
    
    # Initialize components
    state_space = StateSpace()
    environment = CryptoEnvironment(random_seed=42)
    
    # Create agent with warm start
    agent = QLearningAgent(
        state_space=state_space,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=0.1,
        epsilon_end=0.01,
        warm_start=True
    )
    
    # Quick training test
    print("\n🏃‍♂️ Quick training test (50 episodes)...")
    agent.train(environment, num_episodes=50, evaluation_frequency=25)
    
    # Evaluate
    results = agent.evaluate(environment, num_episodes=10)
    
    print("\n✅ Q-Learning test completed!")
