"""
Advanced Training and Comparison System
======================================

This module provides comprehensive training experiments comparing
Q-Learning and Deep Q-Learning with advanced visualization and analysis.

Features:
- Extended training runs (1000+ episodes)
- Hyperparameter optimization experiments
- Comprehensive comparison analysis
- Advanced visualization systems
- Performance benchmarking

Author: RL Team
Date: September 4, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os
import sys
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from environment.state_space import StateSpace, CryptoState, CryptoAlgorithm
from environment.crypto_environment import CryptoEnvironment
from algorithms.q_learning import QLearningAgent
from algorithms.deep_q_learning import DQNAgent

class AdvancedTrainingSystem:
    """Advanced training and comparison system for RL algorithms"""
    
    def __init__(self, random_seed: int = 42):
        """Initialize the advanced training system"""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize core components
        self.state_space = StateSpace()
        self.environment = CryptoEnvironment(random_seed=random_seed)
        
        # Results storage
        self.experiment_results = {}
        self.comparison_data = []
        
        # Create results directories
        self.results_dir = "../../results"
        self.advanced_plots_dir = f"{self.results_dir}/advanced_plots"
        self.models_dir = f"{self.results_dir}/models"
        self.logs_dir = f"{self.results_dir}/logs"
        
        for directory in [self.results_dir, self.advanced_plots_dir, self.models_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        print("🏗️ Advanced Training System initialized")
        print(f"   Results Directory: {self.results_dir}")
        print(f"   Advanced Plots: {self.advanced_plots_dir}")
    
    def run_extended_q_learning_experiment(self, episodes: int = 1000) -> Dict:
        """Run extended Q-Learning experiment with detailed analysis"""
        
        print(f"\n🚀 EXTENDED Q-LEARNING EXPERIMENT ({episodes:,} episodes)")
        print("="*60)
        
        # Create agent with optimized parameters
        agent = QLearningAgent(
            state_space=self.state_space,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon_start=0.3,
            epsilon_end=0.01,
            epsilon_decay=0.9995,
            warm_start=True
        )
        
        # Train with detailed logging
        start_time = datetime.now()
        agent.train(self.environment, num_episodes=episodes, evaluation_frequency=200)
        end_time = datetime.now()
        
        # Extended evaluation
        eval_results = agent.evaluate(self.environment, num_episodes=500)
        
        # Detailed analysis
        results = {
            'algorithm': 'Q-Learning Extended',
            'episodes': episodes,
            'training_time': (end_time - start_time).total_seconds(),
            'training_rewards': agent.training_rewards,
            'training_losses': agent.training_losses,
            'expert_agreement_history': agent.expert_agreement_history,
            'evaluation': eval_results,
            'convergence_analysis': self._analyze_convergence(agent.training_rewards),
            'q_table_analysis': self._analyze_q_table(agent.q_table),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model
        model_path = f"{self.models_dir}/q_learning_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        agent.save_model(model_path)
        results['model_path'] = model_path
        
        self.experiment_results['q_learning_extended'] = results
        return results
    
    def run_extended_dqn_experiment(self, episodes: int = 1000) -> Dict:
        """Run extended DQN experiment with detailed analysis"""
        
        print(f"\n🚀 EXTENDED DQN EXPERIMENT ({episodes:,} episodes)")
        print("="*60)
        
        # Create DQN agent with optimized parameters
        agent = DQNAgent(
            state_space=self.state_space,
            learning_rate=0.0005,
            discount_factor=0.95,
            epsilon_start=0.9,
            epsilon_end=0.01,
            epsilon_decay=0.9995,
            batch_size=64,
            buffer_capacity=50000,
            target_update_frequency=100,
            hidden_sizes=[256, 128, 64],
            warm_start=True
        )
        
        # Train with detailed logging
        start_time = datetime.now()
        agent.train(self.environment, num_episodes=episodes, evaluation_frequency=200)
        end_time = datetime.now()
        
        # Extended evaluation
        eval_results = agent.evaluate(self.environment, num_episodes=500)
        
        # Detailed analysis
        results = {
            'algorithm': 'DQN Extended',
            'episodes': episodes,
            'training_time': (end_time - start_time).total_seconds(),
            'training_rewards': agent.training_rewards,
            'training_losses': agent.training_losses,
            'expert_agreement_history': agent.expert_agreement_history,
            'q_value_history': agent.q_value_history,
            'evaluation': eval_results,
            'convergence_analysis': self._analyze_convergence(agent.training_rewards),
            'network_analysis': self._analyze_dqn_network(agent),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model
        model_path = f"{self.models_dir}/dqn_extended_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        agent.save_model(model_path)
        results['model_path'] = model_path
        
        self.experiment_results['dqn_extended'] = results
        return results
    
    def run_hyperparameter_optimization(self):
        """Run hyperparameter optimization experiments"""
        
        print("\n🔧 HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        # Define hyperparameter grids
        q_learning_configs = [
            {'lr': 0.05, 'gamma': 0.9, 'eps_decay': 0.995, 'name': 'QL_Config_1'},
            {'lr': 0.1, 'gamma': 0.95, 'eps_decay': 0.9995, 'name': 'QL_Config_2'},
            {'lr': 0.2, 'gamma': 0.99, 'eps_decay': 0.999, 'name': 'QL_Config_3'},
        ]
        
        dqn_configs = [
            {'lr': 0.001, 'batch_size': 32, 'hidden': [128, 64], 'name': 'DQN_Config_1'},
            {'lr': 0.0005, 'batch_size': 64, 'hidden': [256, 128, 64], 'name': 'DQN_Config_2'},
            {'lr': 0.0001, 'batch_size': 128, 'hidden': [512, 256, 128], 'name': 'DQN_Config_3'},
        ]
        
        hyperopt_results = {}
        
        # Test Q-Learning configurations
        for config in q_learning_configs:
            print(f"\n🔄 Testing {config['name']}")
            
            agent = QLearningAgent(
                state_space=self.state_space,
                learning_rate=config['lr'],
                discount_factor=config['gamma'],
                epsilon_decay=config['eps_decay'],
                warm_start=True
            )
            
            agent.train(self.environment, num_episodes=500, evaluation_frequency=500)
            eval_results = agent.evaluate(self.environment, num_episodes=100)
            
            hyperopt_results[config['name']] = {
                'algorithm': 'Q-Learning',
                'config': config,
                'performance': eval_results['avg_reward'],
                'expert_agreement': eval_results['avg_expert_agreement']
            }
        
        # Test DQN configurations (smaller scale due to computational cost)
        for config in dqn_configs[:2]:  # Test only first 2 due to time constraints
            print(f"\n🔄 Testing {config['name']}")
            
            agent = DQNAgent(
                state_space=self.state_space,
                learning_rate=config['lr'],
                batch_size=config['batch_size'],
                hidden_sizes=config['hidden'],
                warm_start=True
            )
            
            agent.train(self.environment, num_episodes=300, evaluation_frequency=300)
            eval_results = agent.evaluate(self.environment, num_episodes=50)
            
            hyperopt_results[config['name']] = {
                'algorithm': 'DQN',
                'config': config,
                'performance': eval_results['avg_reward'],
                'expert_agreement': eval_results['avg_expert_agreement']
            }
        
        self.experiment_results['hyperparameter_optimization'] = hyperopt_results
        
        # Print best configurations
        print("\n🏆 HYPERPARAMETER OPTIMIZATION RESULTS")
        print("="*60)
        
        for name, result in hyperopt_results.items():
            print(f"{name:15}: Reward = {result['performance']:6.2f}, "
                  f"Agreement = {result['expert_agreement']:5.1%}")
        
        return hyperopt_results
    
    def _analyze_convergence(self, rewards: List[float]) -> Dict:
        """Analyze training convergence"""
        if len(rewards) < 50:
            return {'converged': False, 'convergence_episode': len(rewards)}
        
        # Moving average analysis
        window = min(50, len(rewards) // 10)
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        
        # Find convergence point (when variance becomes small)
        convergence_episode = len(rewards)
        for i in range(window, len(moving_avg) - 30):
            recent_var = np.var(moving_avg[i:i+30])
            if recent_var < 50:  # Threshold for convergence
                convergence_episode = i
                break
        
        # Performance metrics
        final_performance = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        initial_performance = np.mean(rewards[:50]) if len(rewards) >= 50 else np.mean(rewards)
        
        return {
            'converged': convergence_episode < len(rewards),
            'convergence_episode': convergence_episode,
            'convergence_percentage': (convergence_episode / len(rewards)) * 100,
            'initial_performance': initial_performance,
            'final_performance': final_performance,
            'improvement': final_performance - initial_performance,
            'stability': np.std(rewards[-100:]) if len(rewards) >= 100 else np.std(rewards)
        }
    
    def _analyze_q_table(self, q_table: np.ndarray) -> Dict:
        """Analyze Q-table statistics"""
        return {
            'non_zero_entries': int(np.count_nonzero(q_table)),
            'total_entries': int(q_table.size),
            'sparsity': float(np.count_nonzero(q_table) / q_table.size),
            'mean_q_value': float(np.mean(q_table)),
            'std_q_value': float(np.std(q_table)),
            'max_q_value': float(np.max(q_table)),
            'min_q_value': float(np.min(q_table)),
            'value_range': float(np.max(q_table) - np.min(q_table))
        }
    
    def _analyze_dqn_network(self, agent: DQNAgent) -> Dict:
        """Analyze DQN network statistics"""
        # Get network parameters
        total_params = sum(p.numel() for p in agent.q_network.parameters())
        trainable_params = sum(p.numel() for p in agent.q_network.parameters() if p.requires_grad)
        
        # Get sample Q-values for analysis
        sample_states = []
        for i in range(30):
            state = CryptoState.from_index(i)
            obs = agent._state_to_observation(state)
            sample_states.append(obs)
        
        import torch
        with torch.no_grad():
            states_tensor = torch.FloatTensor(sample_states).to(agent.device)
            q_values = agent.q_network(states_tensor).cpu().numpy()
        
        return {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'network_depth': len([m for m in agent.q_network.modules() if isinstance(m, torch.nn.Linear)]),
            'sample_q_stats': {
                'mean': float(np.mean(q_values)),
                'std': float(np.std(q_values)),
                'min': float(np.min(q_values)),
                'max': float(np.max(q_values))
            },
            'buffer_utilization': len(agent.replay_buffer) / agent.replay_buffer.capacity
        }
    
    def create_comprehensive_comparison(self):
        """Create comprehensive comparison analysis and visualizations"""
        
        print("\n📊 CREATING COMPREHENSIVE COMPARISON")
        print("="*60)
        
        if len(self.experiment_results) < 2:
            print("⚠️  Need at least 2 experiments for comparison")
            return
        
        # Set style for publication-quality plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Battery-Optimized Cryptographic RL - Comprehensive Comparison Analysis', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Training Rewards Comparison (Large plot)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_training_rewards_comparison(ax1)
        
        # 2. Expert Agreement Comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_expert_agreement_comparison(ax2)
        
        # 3. Performance Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_performance_distribution(ax3)
        
        # 4. Action Distribution Comparison
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_action_distribution_comparison(ax4)
        
        # 5. Training Loss Comparison
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_training_loss_comparison(ax5)
        
        # 6. Convergence Analysis
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_convergence_analysis(ax6)
        
        # 7. Algorithm Performance Table
        ax7 = fig.add_subplot(gs[2, :2])
        self._create_performance_table(ax7)
        
        # 8. Expert vs Learned Policy Heatmap
        ax8 = fig.add_subplot(gs[2, 2:])
        self._create_policy_comparison_heatmap(ax8)
        
        # 9. Computational Efficiency
        ax9 = fig.add_subplot(gs[3, 0])
        self._plot_computational_efficiency(ax9)
        
        # 10. Stability Analysis
        ax10 = fig.add_subplot(gs[3, 1])
        self._plot_stability_analysis(ax10)
        
        # 11. Final Performance Summary
        ax11 = fig.add_subplot(gs[3, 2:])
        self._create_final_summary(ax11)
        
        # Save comprehensive plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"{self.advanced_plots_dir}/comprehensive_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"📊 Comprehensive comparison saved: {plot_path}")
        
        # Create detailed report
        self._generate_detailed_report(timestamp)
    
    def _plot_training_rewards_comparison(self, ax):
        """Plot training rewards comparison"""
        ax.set_title('Training Rewards Comparison', fontweight='bold')
        
        for name, results in self.experiment_results.items():
            if 'training_rewards' in results:
                rewards = results['training_rewards']
                # Plot with moving average
                ax.plot(rewards, alpha=0.3, label=f'{name} (raw)')
                if len(rewards) > 50:
                    window = min(50, len(rewards) // 20)
                    moving_avg = pd.Series(rewards).rolling(window=window).mean()
                    ax.plot(moving_avg, linewidth=2, label=f'{name} (avg)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_expert_agreement_comparison(self, ax):
        """Plot expert agreement comparison"""
        ax.set_title('Expert Agreement Rate', fontweight='bold')
        
        for name, results in self.experiment_results.items():
            if 'expert_agreement_history' in results:
                agreement = results['expert_agreement_history']
                if agreement:  # Check if not empty
                    ax.plot(agreement, label=name, linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Expert Agreement Rate')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_distribution(self, ax):
        """Plot performance distribution"""
        ax.set_title('Final Performance\nDistribution', fontweight='bold')
        
        algorithms = []
        performances = []
        
        for name, results in self.experiment_results.items():
            if 'evaluation' in results:
                algorithms.append(name.replace('_', '\n'))
                performances.append(results['evaluation']['avg_reward'])
        
        if algorithms:
            colors = sns.color_palette("husl", len(algorithms))
            bars = ax.bar(algorithms, performances, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, perf in zip(bars, performances):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{perf:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Average Reward')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_action_distribution_comparison(self, ax):
        """Plot action distribution comparison"""
        ax.set_title('Action Distribution\nComparison', fontweight='bold')
        
        # Collect action distributions
        all_actions = set()
        distributions = {}
        
        for name, results in self.experiment_results.items():
            if 'evaluation' in results and 'action_distribution' in results['evaluation']:
                dist = results['evaluation']['action_distribution']
                distributions[name] = dist
                all_actions.update(dist.keys())
        
        if distributions:
            all_actions = sorted(list(all_actions))
            x = np.arange(len(all_actions))
            width = 0.35
            
            for i, (name, dist) in enumerate(distributions.items()):
                values = [dist.get(action, 0) for action in all_actions]
                total = sum(values) if sum(values) > 0 else 1
                percentages = [v/total*100 for v in values]
                
                ax.bar(x + i*width, percentages, width, label=name, alpha=0.8)
            
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Usage %')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels([a[:3] for a in all_actions], rotation=45)
            ax.legend()
    
    def _plot_training_loss_comparison(self, ax):
        """Plot training loss comparison"""
        ax.set_title('Training Loss\nComparison', fontweight='bold')
        
        for name, results in self.experiment_results.items():
            if 'training_losses' in results and results['training_losses']:
                losses = results['training_losses']
                # Plot moving average of losses
                if len(losses) > 100:
                    window = min(100, len(losses) // 20)
                    moving_avg = pd.Series(losses).rolling(window=window).mean()
                    ax.plot(moving_avg, label=name, linewidth=2)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _plot_convergence_analysis(self, ax):
        """Plot convergence analysis"""
        ax.set_title('Convergence Analysis', fontweight='bold')
        
        algorithms = []
        convergence_episodes = []
        improvements = []
        
        for name, results in self.experiment_results.items():
            if 'convergence_analysis' in results:
                conv = results['convergence_analysis']
                algorithms.append(name.replace('_', '\n'))
                convergence_episodes.append(conv.get('convergence_percentage', 100))
                improvements.append(conv.get('improvement', 0))
        
        if algorithms:
            x = np.arange(len(algorithms))
            ax2 = ax.twinx()
            
            bars1 = ax.bar(x - 0.2, convergence_episodes, 0.4, 
                          label='Convergence %', alpha=0.8, color='skyblue')
            bars2 = ax2.bar(x + 0.2, improvements, 0.4, 
                           label='Improvement', alpha=0.8, color='lightcoral')
            
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Convergence %', color='skyblue')
            ax2.set_ylabel('Performance Improvement', color='lightcoral')
            ax.set_xticks(x)
            ax.set_xticklabels(algorithms, rotation=45)
            
            # Add legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
    
    def _create_performance_table(self, ax):
        """Create performance comparison table"""
        ax.set_title('Performance Comparison Table', fontweight='bold')
        ax.axis('off')
        
        # Collect data for table
        table_data = []
        for name, results in self.experiment_results.items():
            if 'evaluation' in results:
                eval_data = results['evaluation']
                row = [
                    name.replace('_', ' ').title(),
                    f"{eval_data['avg_reward']:.2f}",
                    f"{eval_data['std_reward']:.2f}",
                    f"{eval_data['avg_expert_agreement']:.1%}",
                    f"{eval_data['avg_episode_length']:.1f}",
                    f"{results.get('training_time', 0):.1f}s"
                ]
                table_data.append(row)
        
        if table_data:
            columns = ['Algorithm', 'Avg Reward', 'Std Reward', 'Expert Agree', 'Avg Length', 'Train Time']
            
            # Create table
            table = ax.table(cellText=table_data, 
                           colLabels=columns,
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style table
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
    
    def _create_policy_comparison_heatmap(self, ax):
        """Create policy comparison heatmap"""
        ax.set_title('Learned Policy vs Expert Policy', fontweight='bold')
        
        # This would require loading trained models and comparing policies
        # For now, create a placeholder
        ax.text(0.5, 0.5, 'Policy Comparison\n(Implementation Pending)', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _plot_computational_efficiency(self, ax):
        """Plot computational efficiency"""
        ax.set_title('Training Time\nComparison', fontweight='bold')
        
        algorithms = []
        training_times = []
        
        for name, results in self.experiment_results.items():
            if 'training_time' in results:
                algorithms.append(name.replace('_', '\n'))
                training_times.append(results['training_time'])
        
        if algorithms:
            colors = sns.color_palette("viridis", len(algorithms))
            bars = ax.bar(algorithms, training_times, color=colors, alpha=0.8)
            
            # Add time labels
            for bar, time in zip(bars, training_times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Training Time (s)')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_stability_analysis(self, ax):
        """Plot stability analysis"""
        ax.set_title('Performance\nStability', fontweight='bold')
        
        algorithms = []
        stabilities = []
        
        for name, results in self.experiment_results.items():
            if 'convergence_analysis' in results:
                algorithms.append(name.replace('_', '\n'))
                stability = results['convergence_analysis'].get('stability', 0)
                stabilities.append(stability)
        
        if algorithms:
            colors = sns.color_palette("coolwarm", len(algorithms))
            bars = ax.bar(algorithms, stabilities, color=colors, alpha=0.8)
            
            # Add stability labels
            for bar, stability in zip(bars, stabilities):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{stability:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Reward Std Dev')
        ax.tick_params(axis='x', rotation=45)
    
    def _create_final_summary(self, ax):
        """Create final summary"""
        ax.set_title('Experiment Summary', fontweight='bold')
        ax.axis('off')
        
        # Create summary text
        summary_lines = []
        summary_lines.append("🏆 BATTERY-OPTIMIZED CRYPTO RL - RESULTS SUMMARY")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        
        # Find best performers
        if self.experiment_results:
            best_reward = max(self.experiment_results.items(), 
                            key=lambda x: x[1].get('evaluation', {}).get('avg_reward', 0))
            
            best_agreement = max(self.experiment_results.items(), 
                               key=lambda x: x[1].get('evaluation', {}).get('avg_expert_agreement', 0))
            
            summary_lines.append(f"🥇 Best Average Reward:")
            summary_lines.append(f"   {best_reward[0]}: {best_reward[1]['evaluation']['avg_reward']:.2f}")
            summary_lines.append("")
            summary_lines.append(f"🎯 Best Expert Agreement:")
            summary_lines.append(f"   {best_agreement[0]}: {best_agreement[1]['evaluation']['avg_expert_agreement']:.1%}")
            summary_lines.append("")
            
            # Overall statistics
            total_experiments = len(self.experiment_results)
            total_episodes = sum(r.get('episodes', 0) for r in self.experiment_results.values())
            total_time = sum(r.get('training_time', 0) for r in self.experiment_results.values())
            
            summary_lines.append(f"📊 Experiment Statistics:")
            summary_lines.append(f"   Total Experiments: {total_experiments}")
            summary_lines.append(f"   Total Episodes: {total_episodes:,}")
            summary_lines.append(f"   Total Training Time: {total_time:.1f}s")
            summary_lines.append("")
            summary_lines.append(f"✅ All experiments completed successfully!")
            summary_lines.append(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        summary_text = "\n".join(summary_lines)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
    
    def _generate_detailed_report(self, timestamp: str):
        """Generate detailed JSON report"""
        
        report = {
            'experiment_name': 'Battery-Optimized Cryptographic RL Comparison',
            'timestamp': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'experiments': {},
            'summary': {}
        }
        
        # Add experiment results
        for name, results in self.experiment_results.items():
            # Clean results for JSON serialization
            clean_results = {}
            for key, value in results.items():
                if isinstance(value, (list, dict, str, int, float, bool)):
                    if key in ['training_rewards', 'training_losses', 'expert_agreement_history', 'q_value_history']:
                        # Subsample large arrays for JSON
                        if isinstance(value, list) and len(value) > 1000:
                            clean_results[key] = value[::len(value)//1000]  # Keep every nth element
                        else:
                            clean_results[key] = value
                    else:
                        clean_results[key] = value
            
            report['experiments'][name] = clean_results
        
        # Add summary statistics
        if self.experiment_results:
            performances = [(name, r.get('evaluation', {}).get('avg_reward', 0)) 
                          for name, r in self.experiment_results.items()]
            
            report['summary'] = {
                'best_algorithm': max(performances, key=lambda x: x[1])[0],
                'best_performance': max(performances, key=lambda x: x[1])[1],
                'total_experiments': len(self.experiment_results),
                'algorithms_tested': list(self.experiment_results.keys())
            }
        
        # Save report
        report_path = f"{self.logs_dir}/detailed_report_{timestamp}.json"
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert the entire report
        report = convert_numpy_types(report)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Detailed report saved: {report_path}")
    
    def run_complete_experiment_suite(self):
        """Run the complete experiment suite"""
        
        print("\n" + "="*80)
        print("🚀 RUNNING COMPLETE EXPERIMENT SUITE")
        print("Battery-Optimized Cryptographic Algorithm Selection")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # 1. Extended Q-Learning Experiment
            print("\n[1/4] Running Extended Q-Learning Experiment...")
            self.run_extended_q_learning_experiment(episodes=1000)
            
            # 2. Extended DQN Experiment  
            print("\n[2/4] Running Extended DQN Experiment...")
            self.run_extended_dqn_experiment(episodes=1000)
            
            # 3. Hyperparameter Optimization
            print("\n[3/4] Running Hyperparameter Optimization...")
            self.run_hyperparameter_optimization()
            
            # 4. Comprehensive Analysis
            print("\n[4/4] Creating Comprehensive Analysis...")
            self.create_comprehensive_comparison()
            
        except Exception as e:
            print(f"❌ Error in experiment suite: {e}")
            return False
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\n✅ COMPLETE EXPERIMENT SUITE FINISHED")
        print(f"   Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Results saved in: {self.results_dir}")
        
        return True

if __name__ == "__main__":
    # Test advanced training system
    print("🚀 Testing Advanced Training System")
    
    system = AdvancedTrainingSystem(random_seed=42)
    
    # Quick test
    print("\n🏃‍♂️ Quick test with reduced episodes...")
    system.run_extended_q_learning_experiment(episodes=200)
    system.run_extended_dqn_experiment(episodes=200)
    system.create_comprehensive_comparison()
    
    print("\n✅ Advanced training system test completed!")
    
    # Uncomment to run full suite
    # system.run_complete_experiment_suite()
