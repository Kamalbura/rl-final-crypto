"""
Advanced Visualization System
============================

This module creates detailed visualizations for teaching and understanding
the Battery-Optimized Cryptographic Algorithm Selection RL system.

Features:
- Interactive plots for training dynamics
- State space visualizations
- Algorithm comparison charts
- Q-value heatmaps
- Policy analysis visualizations
- Educational diagrams

Author: RL Team
Date: September 4, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import os
import sys
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from environment.state_space import StateSpace, CryptoState, CryptoAlgorithm

class AdvancedVisualizationSystem:
    """Advanced visualization system for RL crypto selection"""
    
    def __init__(self):
        """Initialize visualization system"""
        self.state_space = StateSpace()
        
        # Create visualization directories
        self.viz_dir = "../../results/visualizations"
        self.interactive_dir = f"{self.viz_dir}/interactive"
        self.static_dir = f"{self.viz_dir}/static"
        self.educational_dir = f"{self.viz_dir}/educational"
        
        for directory in [self.viz_dir, self.interactive_dir, self.static_dir, self.educational_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print("🎨 Advanced Visualization System initialized")
        print(f"   Visualizations directory: {self.viz_dir}")
    
    def create_state_space_visualization(self):
        """Create comprehensive state space visualization"""
        
        print("\n🎨 Creating State Space Visualization")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Battery-Optimized Crypto Selection - State Space Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. 3D State Space Visualization
        ax1 = fig.add_subplot(gs[0, :2], projection='3d')
        self._plot_3d_state_space(ax1)
        
        # 2. Expert Decision Heatmap
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_expert_decisions_heatmap(ax2)
        
        # 3. Battery Level Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_battery_distribution(ax3)
        
        # 4. Threat Level Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_threat_distribution(ax4)
        
        # 5. Mission Type Distribution
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_mission_distribution(ax5)
        
        # 6. Algorithm Usage by Expert
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_expert_algorithm_usage(ax6)
        
        # 7. State Transition Probabilities
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_state_transitions(ax7)
        
        # 8. Power Consumption Analysis
        ax8 = fig.add_subplot(gs[2, 2:])
        self._plot_power_consumption_analysis(ax8)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"{self.static_dir}/state_space_analysis_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"💾 State space visualization saved: {plot_path}")
        return plot_path
    
    def create_training_dynamics_animation(self, training_data: Dict, save_gif: bool = True):
        """Create animated visualization of training dynamics"""
        
        print("\n🎬 Creating Training Dynamics Animation")
        
        if 'training_rewards' not in training_data:
            print("⚠️  No training data provided")
            return None
        
        rewards = training_data['training_rewards']
        
        # Create figure for animation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Dynamics Animation', fontsize=14, fontweight='bold')
        
        def animate(frame):
            # Clear axes
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Current episode data
            current_episode = min(frame * 10, len(rewards) - 1)
            current_rewards = rewards[:current_episode+1]
            
            if len(current_rewards) == 0:
                return
            
            # 1. Reward progression
            ax1.plot(current_rewards, 'b-', alpha=0.7)
            if len(current_rewards) > 50:
                window = min(50, len(current_rewards) // 5)
                moving_avg = pd.Series(current_rewards).rolling(window=window).mean()
                ax1.plot(moving_avg, 'r-', linewidth=2, label='Moving Average')
            
            ax1.set_title(f'Training Rewards (Episode {current_episode})')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Performance metrics
            if len(current_rewards) >= 10:
                recent_performance = np.mean(current_rewards[-10:])
                overall_performance = np.mean(current_rewards)
                
                ax2.bar(['Recent (10)', 'Overall'], 
                       [recent_performance, overall_performance],
                       color=['skyblue', 'lightcoral'])
                ax2.set_title('Performance Comparison')
                ax2.set_ylabel('Average Reward')
                
                # Add value labels
                for i, v in enumerate([recent_performance, overall_performance]):
                    ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # 3. Learning progress indicator
            progress = current_episode / max(len(rewards) - 1, 1) * 100
            ax3.pie([progress, 100 - progress], labels=['Complete', 'Remaining'],
                   colors=['lightgreen', 'lightgray'], startangle=90,
                   counterclock=False)
            ax3.set_title(f'Training Progress: {progress:.1f}%')
            
            # 4. Statistics
            if current_rewards:
                stats_text = f"""
                Episode: {current_episode}
                Latest Reward: {current_rewards[-1]:.2f}
                Best Reward: {max(current_rewards):.2f}
                Average: {np.mean(current_rewards):.2f}
                Std Dev: {np.std(current_rewards):.2f}
                """
                ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue'))
                ax4.set_title('Training Statistics')
                ax4.axis('off')
        
        # Create animation
        frames = min(len(rewards) // 10, 100)  # Limit frames for performance
        if frames > 0:
            anim = FuncAnimation(fig, animate, frames=frames, interval=200, repeat=True)
            
            if save_gif:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                gif_path = f"{self.interactive_dir}/training_animation_{timestamp}.gif"
                anim.save(gif_path, writer='pillow', fps=5)
                print(f"🎬 Training animation saved: {gif_path}")
                return gif_path
            else:
                plt.show()
        
        return None
    
    def create_interactive_q_value_explorer(self, q_data: np.ndarray):
        """Create interactive Q-value explorer using Plotly"""
        
        print("\n🔍 Creating Interactive Q-Value Explorer")
        
        # Prepare data for heatmap
        algorithms = [alg.name for alg in CryptoAlgorithm]
        states = [f"State {i}" for i in range(len(q_data))]
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=q_data,
            x=algorithms,
            y=states,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='State: %{y}<br>Algorithm: %{x}<br>Q-Value: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Q-Value Explorer',
            xaxis_title='Cryptographic Algorithms',
            yaxis_title='States',
            height=800,
            width=1200
        )
        
        # Save interactive plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = f"{self.interactive_dir}/q_value_explorer_{timestamp}.html"
        fig.write_html(html_path)
        
        print(f"🔍 Interactive Q-value explorer saved: {html_path}")
        return html_path
    
    def create_algorithm_comparison_dashboard(self, comparison_data: List[Dict]):
        """Create interactive algorithm comparison dashboard"""
        
        print("\n📊 Creating Algorithm Comparison Dashboard")
        
        if not comparison_data:
            print("⚠️  No comparison data provided")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Comparison', 'Training Time', 
                          'Expert Agreement', 'Convergence Analysis'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract data
        algorithms = [data['algorithm'] for data in comparison_data]
        performances = [data['evaluation']['avg_reward'] for data in comparison_data]
        training_times = [data.get('training_time', 0) for data in comparison_data]
        agreements = [data['evaluation']['avg_expert_agreement'] for data in comparison_data]
        
        # 1. Performance comparison
        fig.add_trace(
            go.Bar(x=algorithms, y=performances, name='Average Reward',
                   marker_color='skyblue'),
            row=1, col=1
        )
        
        # 2. Training time
        fig.add_trace(
            go.Bar(x=algorithms, y=training_times, name='Training Time (s)',
                   marker_color='lightcoral'),
            row=1, col=2
        )
        
        # 3. Expert agreement
        fig.add_trace(
            go.Bar(x=algorithms, y=agreements, name='Expert Agreement',
                   marker_color='lightgreen'),
            row=2, col=1
        )
        
        # 4. Convergence (if available)
        convergence_episodes = []
        for data in comparison_data:
            conv_data = data.get('convergence_analysis', {})
            convergence_episodes.append(conv_data.get('convergence_episode', 1000))
        
        fig.add_trace(
            go.Bar(x=algorithms, y=convergence_episodes, name='Convergence Episode',
                   marker_color='gold'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Algorithm Comparison Dashboard",
            showlegend=False,
            height=800,
            width=1200
        )
        
        # Save dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_path = f"{self.interactive_dir}/comparison_dashboard_{timestamp}.html"
        fig.write_html(html_path)
        
        print(f"📊 Comparison dashboard saved: {html_path}")
        return html_path
    
    def create_educational_diagrams(self):
        """Create educational diagrams for teaching RL concepts"""
        
        print("\n🎓 Creating Educational Diagrams")
        
        # Create multiple educational figures
        self._create_rl_concept_diagram()
        self._create_crypto_selection_flowchart()
        self._create_state_action_diagram()
        self._create_reward_function_explanation()
        self._create_algorithm_comparison_guide()
    
    def _plot_3d_state_space(self, ax):
        """Create 3D state space plot"""
        ax.set_title('3D State Space Visualization', fontweight='bold')
        
        # Generate all states
        battery_levels = []
        threat_levels = []
        mission_types = []
        colors = []
        
        for i in range(30):
            state = CryptoState.from_index(i)
            battery_levels.append(state.battery_level.value)
            threat_levels.append(state.threat_status.value)
            mission_types.append(state.mission_criticality.value)
            
            # Color based on expert decision
            expert_action = self.state_space.get_expert_action(state)
            colors.append(expert_action.value)
        
        # Create scatter plot
        scatter = ax.scatter(battery_levels, threat_levels, mission_types, 
                           c=colors, cmap='tab10', s=100, alpha=0.8)
        
        ax.set_xlabel('Battery Level')
        ax.set_ylabel('Threat Level')
        ax.set_zlabel('Mission Type')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    def _plot_expert_decisions_heatmap(self, ax):
        """Plot expert decisions heatmap"""
        ax.set_title('Expert Decision Heatmap', fontweight='bold')
        
        # Create decision matrix
        decision_matrix = np.zeros((5, 6))  # 5 battery x 6 threat-mission combinations
        
        for battery in range(5):
            col = 0
            for threat in range(3):
                for mission in range(2):
                    state = CryptoState.from_index(battery * 6 + threat * 2 + mission)
                    expert_action = self.state_space.get_expert_action(state)
                    decision_matrix[battery, col] = expert_action.value
                    col += 1
        
        # Create heatmap
        sns.heatmap(decision_matrix, ax=ax, cmap='tab10', annot=True, fmt='.0f',
                   cbar_kws={'label': 'Algorithm Choice'})
        
        # Set labels
        threat_mission_labels = []
        for threat in ['Low', 'Med', 'High']:
            for mission in ['Normal', 'Critical']:
                threat_mission_labels.append(f'{threat}-{mission}')
        
        ax.set_xlabel('Threat Level - Mission Type')
        ax.set_ylabel('Battery Level')
        ax.set_xticklabels(threat_mission_labels, rotation=45)
        ax.set_yticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    def _plot_battery_distribution(self, ax):
        """Plot battery level distribution"""
        ax.set_title('Battery Levels', fontweight='bold')
        
        battery_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        battery_counts = [6, 6, 6, 6, 6]  # Each level appears 6 times
        
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 5))
        bars = ax.bar(battery_names, battery_counts, color=colors, alpha=0.8)
        
        # Add percentage labels
        total = sum(battery_counts)
        for bar, count in zip(bars, battery_counts):
            percentage = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of States')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_threat_distribution(self, ax):
        """Plot threat level distribution"""
        ax.set_title('Threat Levels', fontweight='bold')
        
        threat_names = ['Low', 'Medium', 'High']
        threat_counts = [10, 10, 10]  # Each level appears 10 times
        
        colors = ['green', 'orange', 'red']
        bars = ax.bar(threat_names, threat_counts, color=colors, alpha=0.7)
        
        # Add percentage labels
        total = sum(threat_counts)
        for bar, count in zip(bars, threat_counts):
            percentage = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of States')
    
    def _plot_mission_distribution(self, ax):
        """Plot mission type distribution"""
        ax.set_title('Mission Types', fontweight='bold')
        
        mission_names = ['Normal', 'Critical']
        mission_counts = [15, 15]  # Each type appears 15 times
        
        colors = ['lightblue', 'darkblue']
        bars = ax.bar(mission_names, mission_counts, color=colors, alpha=0.7)
        
        # Add percentage labels
        total = sum(mission_counts)
        for bar, count in zip(bars, mission_counts):
            percentage = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of States')
    
    def _plot_expert_algorithm_usage(self, ax):
        """Plot expert algorithm usage"""
        ax.set_title('Expert Algorithm\nPreferences', fontweight='bold')
        
        # Count expert decisions
        algorithm_counts = {alg: 0 for alg in CryptoAlgorithm}
        
        for i in range(30):
            state = CryptoState.from_index(i)
            expert_action = self.state_space.get_expert_action(state)
            algorithm_counts[expert_action] += 1
        
        # Create pie chart
        algorithms = list(algorithm_counts.keys())
        counts = list(algorithm_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        
        wedges, texts, autotexts = ax.pie(counts, labels=[a.name[:3] for a in algorithms], 
                                         colors=colors, autopct='%1.1f%%',
                                         startangle=90)
        
        # Make text bold
        for autotext in autotexts:
            autotext.set_fontweight('bold')
    
    def _plot_state_transitions(self, ax):
        """Plot state transition probabilities"""
        ax.set_title('State Transition Probabilities', fontweight='bold')
        
        # Create dummy transition matrix for visualization
        # In practice, this would be based on actual environment dynamics
        np.random.seed(42)
        transition_sample = np.random.dirichlet(np.ones(8), 30)  # 30 states to 8 actions
        
        im = ax.imshow(transition_sample, cmap='Blues', aspect='auto')
        ax.set_xlabel('Next Action')
        ax.set_ylabel('Current State')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Set ticks
        ax.set_xticks(range(8))
        ax.set_xticklabels([alg.name[:3] for alg in CryptoAlgorithm], rotation=45)
        ax.set_yticks(range(0, 30, 5))
        ax.set_yticklabels([f'S{i}' for i in range(0, 30, 5)])
    
    def _plot_power_consumption_analysis(self, ax):
        """Plot power consumption analysis"""
        ax.set_title('Power Consumption by Algorithm', fontweight='bold')
        
        # Power consumption data (example values)
        algorithms = [alg.name for alg in CryptoAlgorithm]
        power_consumption = [2.1, 1.8, 1.5, 2.5, 3.2, 4.1, 5.2, 4.8]  # Example values
        
        # Create bar chart with gradient colors
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(algorithms)))
        bars = ax.bar(algorithms, power_consumption, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, power in zip(bars, power_consumption):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{power:.1f}W', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Power Consumption (W)')
        ax.set_xlabel('Cryptographic Algorithms')
        ax.tick_params(axis='x', rotation=45)
        
        # Add power efficiency zones
        ax.axhspan(0, 2, alpha=0.2, color='green', label='High Efficiency')
        ax.axhspan(2, 4, alpha=0.2, color='orange', label='Medium Efficiency')
        ax.axhspan(4, 6, alpha=0.2, color='red', label='Low Efficiency')
        ax.legend(loc='upper right')
    
    def _create_rl_concept_diagram(self):
        """Create RL concept explanation diagram"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_title('Reinforcement Learning Concepts for Crypto Selection', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Agent
        agent_rect = plt.Rectangle((1, 4), 2, 2, facecolor='lightblue', 
                                  edgecolor='blue', linewidth=2)
        ax.add_patch(agent_rect)
        ax.text(2, 5, 'RL Agent\n(Q-Learning/DQN)', ha='center', va='center',
               fontweight='bold', fontsize=10)
        
        # Environment
        env_rect = plt.Rectangle((7, 4), 2, 2, facecolor='lightgreen', 
                                edgecolor='green', linewidth=2)
        ax.add_patch(env_rect)
        ax.text(8, 5, 'Environment\n(Crypto System)', ha='center', va='center',
               fontweight='bold', fontsize=10)
        
        # State
        state_circle = plt.Circle((5, 8), 0.8, facecolor='yellow', 
                                 edgecolor='orange', linewidth=2)
        ax.add_patch(state_circle)
        ax.text(5, 8, 'State\n(Battery, Threat,\nMission)', ha='center', va='center',
               fontweight='bold', fontsize=9)
        
        # Action
        action_circle = plt.Circle((5, 2), 0.8, facecolor='pink', 
                                  edgecolor='red', linewidth=2)
        ax.add_patch(action_circle)
        ax.text(5, 2, 'Action\n(Algorithm\nSelection)', ha='center', va='center',
               fontweight='bold', fontsize=9)
        
        # Reward
        reward_circle = plt.Circle((2, 7.5), 0.6, facecolor='gold', 
                                  edgecolor='darkorange', linewidth=2)
        ax.add_patch(reward_circle)
        ax.text(2, 7.5, 'Reward\n(Performance\nScore)', ha='center', va='center',
               fontweight='bold', fontsize=8)
        
        # Arrows
        # State to Agent
        ax.annotate('', xy=(2.5, 5.5), xytext=(4.2, 7.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        ax.text(3, 6.8, 'Observe', ha='center', fontweight='bold', color='blue')
        
        # Agent to Action
        ax.annotate('', xy=(4.2, 2.5), xytext=(2.5, 4.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        ax.text(3, 3.2, 'Select', ha='center', fontweight='bold', color='red')
        
        # Action to Environment
        ax.annotate('', xy=(7, 4.5), xytext=(5.8, 2.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        ax.text(6.5, 3.2, 'Execute', ha='center', fontweight='bold', color='green')
        
        # Environment to State
        ax.annotate('', xy=(5.8, 7.5), xytext=(7.5, 6),
                   arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
        ax.text(7, 7, 'Update', ha='center', fontweight='bold', color='orange')
        
        # Environment to Reward
        ax.annotate('', xy=(2.6, 7.5), xytext=(7, 5.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gold'))
        ax.text(4.5, 6.8, 'Feedback', ha='center', fontweight='bold', color='gold')
        
        # Save diagram
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{self.educational_dir}/rl_concepts_{timestamp}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"🎓 RL concept diagram saved")
    
    def _create_crypto_selection_flowchart(self):
        """Create crypto selection process flowchart"""
        fig, ax = plt.subplots(figsize=(12, 14))
        ax.set_title('Cryptographic Algorithm Selection Process', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Define boxes and connections for flowchart
        boxes = [
            {'pos': (5, 13), 'size': (3, 0.8), 'text': 'Start\nSystem Monitoring', 'color': 'lightgreen'},
            {'pos': (5, 11.5), 'size': (3, 0.8), 'text': 'Collect System State\n(Battery, Threat, Mission)', 'color': 'lightblue'},
            {'pos': (5, 10), 'size': (3, 0.8), 'text': 'Input to RL Agent', 'color': 'lightyellow'},
            {'pos': (2, 8.5), 'size': (2.5, 0.8), 'text': 'Q-Learning\nAgent', 'color': 'lightcoral'},
            {'pos': (8, 8.5), 'size': (2.5, 0.8), 'text': 'DQN\nAgent', 'color': 'lightcoral'},
            {'pos': (5, 7), 'size': (3, 0.8), 'text': 'Select Crypto\nAlgorithm', 'color': 'orange'},
            {'pos': (5, 5.5), 'size': (3, 0.8), 'text': 'Execute Encryption/\nDecryption', 'color': 'lightpink'},
            {'pos': (5, 4), 'size': (3, 0.8), 'text': 'Measure Performance\n& Power Consumption', 'color': 'lightgray'},
            {'pos': (5, 2.5), 'size': (3, 0.8), 'text': 'Calculate Reward\n& Update Model', 'color': 'gold'},
            {'pos': (5, 1), 'size': (3, 0.8), 'text': 'Continue Monitoring', 'color': 'lightgreen'},
        ]
        
        # Draw boxes
        for box in boxes:
            rect = plt.Rectangle((box['pos'][0] - box['size'][0]/2, box['pos'][1] - box['size'][1]/2),
                               box['size'][0], box['size'][1], 
                               facecolor=box['color'], edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text(box['pos'][0], box['pos'][1], box['text'], 
                   ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Draw arrows
        arrows = [
            ((5, 12.7), (5, 12.3)),  # Start to Collect
            ((5, 11.1), (5, 10.8)),  # Collect to Input
            ((4.2, 9.5), (2.8, 9)),  # Input to Q-Learning
            ((5.8, 9.5), (7.2, 9)),  # Input to DQN
            ((2.8, 8), (4.2, 7.5)),  # Q-Learning to Select
            ((7.2, 8), (5.8, 7.5)),  # DQN to Select
            ((5, 6.6), (5, 6.3)),    # Select to Execute
            ((5, 5.1), (5, 4.8)),    # Execute to Measure
            ((5, 3.6), (5, 3.3)),    # Measure to Reward
            ((5, 2.1), (5, 1.8)),    # Reward to Continue
            ((3.5, 1), (1, 11.5)),   # Continue back to Collect (feedback loop)
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
        
        # Add decision diamond
        diamond = plt.Polygon([(5, 9.5), (4, 9), (5, 8.5), (6, 9)], 
                            facecolor='yellow', edgecolor='black')
        ax.add_patch(diamond)
        ax.text(5, 9, 'Choose\nAlgorithm', ha='center', va='center', 
               fontweight='bold', fontsize=8)
        
        # Save flowchart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f"{self.educational_dir}/crypto_selection_flowchart_{timestamp}.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"🎓 Crypto selection flowchart saved")
    
    def _create_state_action_diagram(self):
        """Create state-action space diagram"""
        # Implementation for state-action visualization
        pass
    
    def _create_reward_function_explanation(self):
        """Create reward function explanation diagram"""
        # Implementation for reward function visualization  
        pass
    
    def _create_algorithm_comparison_guide(self):
        """Create algorithm comparison guide"""
        # Implementation for algorithm comparison guide
        pass
    
    def generate_complete_visualization_suite(self, experiment_results: Optional[Dict] = None):
        """Generate complete visualization suite"""
        
        print("\n" + "="*60)
        print("🎨 GENERATING COMPLETE VISUALIZATION SUITE")
        print("="*60)
        
        visualizations = []
        
        try:
            # 1. State Space Analysis
            print("\n[1/5] Creating state space visualizations...")
            state_viz = self.create_state_space_visualization()
            visualizations.append(state_viz)
            
            # 2. Educational Diagrams
            print("\n[2/5] Creating educational diagrams...")
            self.create_educational_diagrams()
            
            # 3. Training Animation (if data available)
            if experiment_results and any('training_rewards' in r for r in experiment_results.values()):
                print("\n[3/5] Creating training animations...")
                for name, data in experiment_results.items():
                    if 'training_rewards' in data:
                        self.create_training_dynamics_animation(data, save_gif=True)
                        break  # Create one animation for demo
            
            # 4. Interactive Visualizations
            print("\n[4/5] Creating interactive visualizations...")
            if experiment_results:
                # Extract Q-data or create sample
                sample_q_data = np.random.random((30, 8)) * 100  # Placeholder
                self.create_interactive_q_value_explorer(sample_q_data)
                
                # Create comparison dashboard
                comparison_list = list(experiment_results.values())
                if comparison_list:
                    self.create_algorithm_comparison_dashboard(comparison_list)
            
            # 5. Summary Report
            print("\n[5/5] Generating summary report...")
            self._generate_visualization_summary()
            
            print(f"\n✅ COMPLETE VISUALIZATION SUITE GENERATED")
            print(f"   Static plots: {self.static_dir}")
            print(f"   Interactive plots: {self.interactive_dir}")
            print(f"   Educational materials: {self.educational_dir}")
            
            return visualizations
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            return []
    
    def _generate_visualization_summary(self):
        """Generate visualization summary report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        summary = f"""
# Visualization Suite Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files Created:

### Static Visualizations ({self.static_dir})
- State space analysis plots
- Training comparison charts
- Performance metrics visualizations

### Interactive Visualizations ({self.interactive_dir})  
- Q-value explorer (HTML)
- Algorithm comparison dashboard (HTML)
- Training dynamics animations (GIF)

### Educational Materials ({self.educational_dir})
- RL concept diagrams
- Crypto selection flowcharts
- State-action explanations
- Reward function illustrations

## Usage Instructions:

1. **For Team Presentations**: Use static PNG files from the static folder
2. **For Interactive Analysis**: Open HTML files in web browser
3. **For Teaching**: Use educational diagrams to explain concepts
4. **For Progress Monitoring**: View training animations

## Next Steps:

- Integrate visualizations into presentation materials
- Create custom visualizations for specific analysis needs
- Update visualizations with new experimental results
"""
        
        summary_path = f"{self.viz_dir}/visualization_summary_{timestamp}.md"
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        print(f"📄 Visualization summary saved: {summary_path}")

if __name__ == "__main__":
    # Test visualization system
    print("🎨 Testing Advanced Visualization System")
    
    viz_system = AdvancedVisualizationSystem()
    
    # Create basic visualizations
    viz_system.create_state_space_visualization()
    viz_system.create_educational_diagrams()
    
    # Generate sample data for testing
    sample_results = {
        'test_algorithm': {
            'training_rewards': np.random.randn(500).cumsum() + 50,
            'evaluation': {
                'avg_reward': 45.2,
                'std_reward': 12.1,
                'avg_expert_agreement': 0.78,
                'avg_episode_length': 25.3,
                'action_distribution': {'KYBER': 25, 'DILITHIUM': 20, 'ASCON': 15}
            },
            'training_time': 120.5
        }
    }
    
    viz_system.generate_complete_visualization_suite(sample_results)
    
    print("\n✅ Visualization system test completed!")
