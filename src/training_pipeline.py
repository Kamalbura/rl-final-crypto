"""
Training Pipeline for Battery-Optimized Cryptographic RL
=======================================================

This module provides a complete training pipeline for both Q-Learning
and Deep Q-Learning with comprehensive logging and visualization.

Author: RL Team  
Date: September 4, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import sys
from typing import Dict, List, Any

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from environment.state_space import StateSpace, CryptoState, CryptoAlgorithm
from environment.crypto_environment import CryptoEnvironment
from algorithms.q_learning import QLearningAgent

class TrainingPipeline:
    """Complete training pipeline with logging and visualization"""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the training pipeline"""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize components
        self.state_space = StateSpace()
        self.environment = CryptoEnvironment(random_seed=random_seed)
        
        # Results storage
        self.results = {}
        self.training_logs = []
        
        # Create results directories
        self.results_dir = "../../results"
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/models", exist_ok=True)
        os.makedirs(f"{self.results_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.results_dir}/logs", exist_ok=True)
        
        print("🏗️ Training Pipeline initialized")
        print(f"   Random Seed: {random_seed}")
        print(f"   Results Directory: {self.results_dir}")
    
    def train_q_learning(
        self,
        episodes: int = 1000,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        warm_start: bool = True,
        save_model: bool = True
    ) -> Dict:
        """Train Q-Learning agent with specified parameters"""
        
        print(f"\n🚀 TRAINING Q-LEARNING AGENT")
        print(f"Episodes: {episodes}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Discount Factor: {discount_factor}")
        print(f"Warm Start: {warm_start}")
        
        # Create agent
        agent = QLearningAgent(
            state_space=self.state_space,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            warm_start=warm_start
        )
        
        # Train
        start_time = datetime.now()
        agent.train(self.environment, num_episodes=episodes)
        end_time = datetime.now()
        
        # Evaluate
        eval_results = agent.evaluate(self.environment, num_episodes=200)
        
        # Store results
        results = {
            'algorithm': 'Q-Learning',
            'parameters': {
                'episodes': episodes,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'warm_start': warm_start
            },
            'training_time': (end_time - start_time).total_seconds(),
            'training_rewards': agent.training_rewards,
            'training_losses': agent.training_losses,
            'expert_agreement_history': agent.expert_agreement_history,
            'evaluation': eval_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['q_learning'] = results
        
        # Save model if requested
        if save_model:
            model_path = f"{self.results_dir}/models/q_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            agent.save_model(model_path)
            results['model_path'] = model_path
        
        # Generate visualizations
        self._create_q_learning_plots(agent, results)
        
        return results
    
    def _create_q_learning_plots(self, agent: QLearningAgent, results: Dict):
        """Create visualization plots for Q-Learning results"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        fig.suptitle('Q-Learning Training Results - Battery-Optimized Crypto RL', fontsize=16, fontweight='bold')
        
        # 1. Training Rewards
        axes[0, 0].plot(agent.training_rewards, alpha=0.7, color='blue')
        if len(agent.training_rewards) > 50:
            # Add moving average
            window = min(50, len(agent.training_rewards) // 10)
            moving_avg = pd.Series(agent.training_rewards).rolling(window=window).mean()
            axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'{window}-episode average')
            axes[0, 0].legend()
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Expert Agreement
        if agent.expert_agreement_history:
            axes[0, 1].plot(agent.expert_agreement_history, color='green', alpha=0.8)
            axes[0, 1].set_title('Expert Agreement Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Agreement Rate')
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. TD Error (Learning Progress)
        if agent.training_losses:
            axes[0, 2].plot(agent.training_losses, alpha=0.6, color='orange')
            if len(agent.training_losses) > 100:
                window = min(100, len(agent.training_losses) // 20)
                moving_avg = pd.Series(agent.training_losses).rolling(window=window).mean()
                axes[0, 2].plot(moving_avg, color='red', linewidth=2)
            axes[0, 2].set_title('TD Error (Learning Progress)')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('TD Error')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Q-values Heatmap
        im = axes[1, 0].imshow(agent.q_table.T, cmap='RdYlBu', aspect='auto')
        axes[1, 0].set_title('Q-Values Heatmap')
        axes[1, 0].set_xlabel('State Index')
        axes[1, 0].set_ylabel('Action Index')
        axes[1, 0].set_yticks(range(8))
        axes[1, 0].set_yticklabels([algo.name for algo in CryptoAlgorithm])
        plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Action Distribution in Evaluation
        eval_results = results['evaluation']
        action_dist = eval_results['action_distribution']
        actions = list(action_dist.keys())
        counts = list(action_dist.values())
        
        axes[1, 1].bar(actions, counts, color='skyblue', alpha=0.8)
        axes[1, 1].set_title('Action Distribution (Evaluation)')
        axes[1, 1].set_xlabel('Algorithm')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Performance Summary
        axes[1, 2].axis('off')
        
        # Create summary text
        summary_text = f"""
PERFORMANCE SUMMARY

Training Episodes: {results['parameters']['episodes']:,}
Training Time: {results['training_time']:.1f}s

Final Performance:
• Avg Reward: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}
• Expert Agreement: {eval_results['avg_expert_agreement']:.1%}
• Avg Episode Length: {eval_results['avg_episode_length']:.1f}

Hyperparameters:
• Learning Rate: {results['parameters']['learning_rate']}
• Discount Factor: {results['parameters']['discount_factor']}
• Warm Start: {"✅" if results['parameters']['warm_start'] else "❌"}

Q-Table Stats:
• Non-zero entries: {np.count_nonzero(agent.q_table)}/240
• Max Q-value: {agent.q_table.max():.2f}
• Min Q-value: {agent.q_table.min():.2f}
        """
        
        axes[1, 2].text(0.05, 0.95, summary_text.strip(), transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=9)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = f"{self.results_dir}/plots/q_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training plots saved to: {plot_path}")
    
    def run_full_experiment(self, save_results: bool = True):
        """Run complete experiment with both algorithms"""
        
        print("\n" + "="*70)
        print("🚀 RUNNING COMPLETE RL EXPERIMENT")
        print("Battery-Optimized Cryptographic Algorithm Selection")
        print("="*70)
        
        # Experiment configurations
        configs = [
            # Q-Learning with different configurations
            {
                'name': 'Q-Learning (Warm Start)',
                'method': 'q_learning',
                'episodes': 1000,
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'warm_start': True
            },
            {
                'name': 'Q-Learning (Cold Start)', 
                'method': 'q_learning',
                'episodes': 1000,
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'warm_start': False
            },
            {
                'name': 'Q-Learning (High LR)',
                'method': 'q_learning', 
                'episodes': 1000,
                'learning_rate': 0.3,
                'discount_factor': 0.95,
                'warm_start': True
            }
        ]
        
        experiment_results = {}
        
        for config in configs:
            print(f"\n🔄 Running: {config['name']}")
            
            if config['method'] == 'q_learning':
                results = self.train_q_learning(
                    episodes=config['episodes'],
                    learning_rate=config['learning_rate'],
                    discount_factor=config['discount_factor'],
                    warm_start=config['warm_start']
                )
                experiment_results[config['name']] = results
        
        # Save experiment results
        if save_results:
            results_path = f"{self.results_dir}/logs/experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for name, results in experiment_results.items():
                json_results[name] = {
                    'algorithm': results['algorithm'],
                    'parameters': results['parameters'],
                    'training_time': results['training_time'],
                    'evaluation': results['evaluation'],
                    'timestamp': results['timestamp'],
                    'final_reward': float(np.mean(results['training_rewards'][-100:]) if results['training_rewards'] else 0),
                    'convergence_episode': self._find_convergence_point(results['training_rewards'])
                }
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"💾 Experiment results saved to: {results_path}")
        
        # Print comparison
        self._print_experiment_comparison(experiment_results)
        
        return experiment_results
    
    def _find_convergence_point(self, rewards: List[float]) -> int:
        """Find approximate convergence point in training"""
        if len(rewards) < 100:
            return len(rewards)
        
        # Use moving average to find when reward stabilizes
        window = 50
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        
        # Find where variance becomes small
        for i in range(window, len(moving_avg) - 50):
            recent_var = np.var(moving_avg[i:i+50])
            if recent_var < 10:  # Threshold for "converged"
                return i
        
        return len(rewards)
    
    def _print_experiment_comparison(self, results: Dict):
        """Print comparison table of all experiments"""
        
        print("\n" + "="*80)
        print("📊 EXPERIMENT COMPARISON")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        for name, result in results.items():
            eval_data = result['evaluation']
            comparison_data.append({
                'Experiment': name,
                'Avg Reward': f"{eval_data['avg_reward']:.2f} ± {eval_data['std_reward']:.2f}",
                'Expert Agreement': f"{eval_data['avg_expert_agreement']:.1%}",
                'Training Time': f"{result['training_time']:.1f}s",
                'Episodes': result['parameters']['episodes'],
                'Warm Start': "✅" if result['parameters']['warm_start'] else "❌"
            })
        
        # Print table
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False, justify='left'))
        
        print("="*80)
        
        # Identify best performer
        if len(results) > 1:
            best_reward = max(results.items(), key=lambda x: x[1]['evaluation']['avg_reward'])
            best_agreement = max(results.items(), key=lambda x: x[1]['evaluation']['avg_expert_agreement'])
            
            print(f"\n🏆 Best Reward: {best_reward[0]} ({best_reward[1]['evaluation']['avg_reward']:.2f})")
            print(f"🎯 Best Expert Agreement: {best_agreement[0]} ({best_agreement[1]['evaluation']['avg_expert_agreement']:.1%})")

if __name__ == "__main__":
    # Run training pipeline
    print("🚀 Running Training Pipeline")
    
    pipeline = TrainingPipeline(random_seed=42)
    
    # Quick test
    print("\n🏃‍♂️ Quick Q-Learning test...")
    results = pipeline.train_q_learning(episodes=100, save_model=False)
    
    print(f"\n✅ Pipeline test completed!")
    print(f"Final reward: {results['evaluation']['avg_reward']:.2f}")
    print(f"Expert agreement: {results['evaluation']['avg_expert_agreement']:.1%}")
    
    # Uncomment to run full experiment
    # pipeline.run_full_experiment()
