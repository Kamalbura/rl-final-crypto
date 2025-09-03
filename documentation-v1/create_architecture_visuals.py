#!/usr/bin/env python3
"""
Architecture Visualization Generator
Creates comprehensive diagrams for the RL Crypto System documentation

Author: RL Team
Date: September 4, 2025
Purpose: Generate professional architecture diagrams and charts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Try to import scipy, fall back to simple smoothing if not available
try:
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

class ArchitectureVisualizer:
    """Generate comprehensive architecture documentation visuals"""
    
    def __init__(self):
        """Initialize the visualization system"""
        self.output_dir = Path(__file__).parent / "images"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up professional styling
        plt.style.use('default')
        
        # Define colors for consistency
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#0B4F6C',
            'light': '#F5F5F5',
            'dark': '#2C2C2C'
        }
        
    def generate_all_diagrams(self):
        """Generate all architecture diagrams"""
        print("🎨 Generating Architecture Documentation Visuals...")
        
        # Core architecture diagrams
        self.create_system_overview()
        self.create_state_space_diagram() 
        self.create_action_space_diagram()
        self.create_reward_function_diagram()
        self.create_learning_pipeline_diagram()
        
        print(f"✅ All diagrams saved to: {self.output_dir}")
        
    def create_system_overview(self):
        """Create comprehensive system architecture overview"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, 'RL Crypto Selection System Architecture', 
                fontsize=20, fontweight='bold', ha='center')
        
        # Main components
        components = [
            # Component, x, y, width, height, color, label
            ('Environment', 1, 7, 3.5, 1.5, self.colors['primary'], 'white'),
            ('RL Agents', 5.5, 7, 3.5, 1.5, self.colors['secondary'], 'white'),
            ('State Space', 1, 5, 2, 1.2, self.colors['info'], 'white'),
            ('Action Space', 3.5, 5, 2, 1.2, self.colors['info'], 'white'),
            ('Reward Function', 6, 5, 2.5, 1.2, self.colors['accent'], 'white'),
            ('Training Pipeline', 1, 3, 3, 1.2, self.colors['success'], 'white'),
            ('Monitoring', 5, 3, 3.5, 1.2, self.colors['dark'], 'white'),
            ('Expert Knowledge', 1, 1, 3, 1.2, '#8B4513', 'white'),
            ('Production System', 5, 1, 3.5, 1.2, '#228B22', 'white')
        ]
        
        for name, x, y, w, h, color, text_color in components:
            # Create fancy boxes
            box = FancyBboxPatch((x, y), w, h, 
                               boxstyle="round,pad=0.1",
                               facecolor=color, 
                               edgecolor='black',
                               linewidth=2)
            ax.add_patch(box)
            
            # Add text
            ax.text(x + w/2, y + h/2, name, 
                   fontsize=12, fontweight='bold',
                   ha='center', va='center', color=text_color)
        
        # Add arrows showing relationships
        arrows = [
            # From, To (center coordinates)
            ((2.75, 7), (2, 6.2)),  # Environment → State Space
            ((2.75, 7), (4.5, 6.2)),  # Environment → Action Space  
            ((7.25, 7), (7.25, 6.2)),  # RL Agents → Reward Function
            ((2.5, 5), (2.5, 4.2)),  # State Space → Training
            ((6.75, 5), (6.75, 4.2)),  # Reward → Monitoring
            ((2.5, 3), (2.5, 2.2)),  # Training → Expert Knowledge
            ((6.75, 3), (6.75, 2.2)),  # Monitoring → Production
        ]
        
        for (x1, y1), (x2, y2) in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle='arc3,rad=0',
                                     lw=2, color='black'))
        
        # Add legend
        legend_elements = [
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['primary'], label='Core Environment'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['secondary'], label='Learning Agents'),  
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['accent'], label='Reward System'),
            mpatches.Rectangle((0, 0), 1, 1, facecolor=self.colors['success'], label='Training Process'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '01_system_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_state_space_diagram(self):
        """Create detailed state space visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main title
        fig.suptitle('State Space Architecture (30 Total States)', fontsize=16, fontweight='bold')
        
        # 1. 3D State Space Visualization
        ax1.set_title('3D State Space Mapping', fontweight='bold')
        
        # Create 3D-like visualization using 2D projection
        battery_levels = ['CRITICAL', 'LOW', 'MEDIUM', 'HIGH', 'FULL']
        threat_levels = ['LOW', 'MEDIUM', 'HIGH'] 
        mission_types = ['NORMAL', 'CRITICAL']
        
        y_pos = 0
        colors_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, battery in enumerate(battery_levels):
            for j, threat in enumerate(threat_levels):
                for k, mission in enumerate(mission_types):
                    state_id = i * 6 + j * 2 + k
                    
                    # Draw state box
                    rect = Rectangle((j*2 + k*0.8, y_pos), 0.7, 0.8, 
                                   facecolor=colors_list[i], alpha=0.7,
                                   edgecolor='black')
                    ax1.add_patch(rect)
                    
                    # Add state ID
                    ax1.text(j*2 + k*0.8 + 0.35, y_pos + 0.4, str(state_id),
                            ha='center', va='center', fontweight='bold')
            y_pos += 1
            
        ax1.set_xlim(-0.5, 6.5)
        ax1.set_ylim(-0.5, 5.5)
        ax1.set_xlabel('Threat Level & Mission Type')
        ax1.set_ylabel('Battery Level')
        ax1.set_xticks([0.4, 1.2, 2.4, 3.2, 4.4, 5.2])
        ax1.set_xticklabels(['L-N', 'L-C', 'M-N', 'M-C', 'H-N', 'H-C'], rotation=45)
        ax1.set_yticks(range(5))
        ax1.set_yticklabels(battery_levels)
        
        # 2. State Distribution Heatmap
        ax2.set_title('State ID Distribution Matrix', fontweight='bold')
        
        state_matrix = np.zeros((5, 6))
        for i in range(5):  # Battery levels
            for j in range(3):  # Threat levels  
                for k in range(2):  # Mission types
                    state_id = i * 6 + j * 2 + k
                    state_matrix[i, j*2 + k] = state_id
        
        im = ax2.imshow(state_matrix, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(6))
        ax2.set_xticklabels(['L-N', 'L-C', 'M-N', 'M-C', 'H-N', 'H-C'])
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(battery_levels)
        ax2.set_xlabel('Threat-Mission Combination')
        ax2.set_ylabel('Battery Level')
        
        # Add state IDs as text
        for i in range(5):
            for j in range(6):
                ax2.text(j, i, f'{int(state_matrix[i,j])}',
                        ha='center', va='center', color='white', fontweight='bold')
        
        plt.colorbar(im, ax=ax2, label='State ID')
        
        # 3. State Complexity Analysis
        ax3.set_title('State Space Complexity', fontweight='bold')
        
        dimensions = ['Battery\nLevels', 'Threat\nLevels', 'Mission\nTypes', 'Total\nStates']
        values = [5, 3, 2, 30]
        colors_dim = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['accent'], self.colors['success']]
        
        bars = ax3.bar(dimensions, values, color=colors_dim, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Count')
        ax3.set_title('Dimensionality Breakdown')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(value), ha='center', va='bottom', fontweight='bold')
        
        # 4. Example State Scenarios
        ax4.set_title('Critical State Examples', fontweight='bold')
        ax4.axis('off')
        
        examples = [
            "State 0: CRITICAL Battery + LOW Threat + NORMAL Mission",
            "→ Expert Choice: SPECK (Ultra-low power)",
            "",
            "State 14: MEDIUM Battery + HIGH Threat + NORMAL Mission", 
            "→ Expert Choice: KYBER (Balanced security/power)",
            "",
            "State 29: FULL Battery + HIGH Threat + CRITICAL Mission",
            "→ Expert Choice: FALCON (Maximum security)",
            "",
            "Key Insights:",
            "• Battery level drives power-conscious decisions",
            "• Threat level determines security requirements", 
            "• Mission criticality affects risk tolerance",
            "• Expert knowledge guides optimal choices"
        ]
        
        y_text = 0.95
        for line in examples:
            if line.startswith("State"):
                color = self.colors['primary']
                weight = 'bold'
            elif line.startswith("→"):
                color = self.colors['accent']
                weight = 'normal'
            elif line.startswith("Key"):
                color = self.colors['success'] 
                weight = 'bold'
            elif line.startswith("•"):
                color = 'black'
                weight = 'normal'
            else:
                color = 'black'
                weight = 'normal'
                
            ax4.text(0.05, y_text, line, transform=ax4.transAxes,
                    fontsize=10, color=color, fontweight=weight)
            y_text -= 0.08
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '02_state_space_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_action_space_diagram(self):
        """Create comprehensive action space visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Action Space: 8 Cryptographic Algorithms', fontsize=16, fontweight='bold')
        
        # Algorithm data
        algorithms = ['ASCON', 'SPECK', 'HIGHT', 'CAMELLIA', 'KYBER', 'DILITHIUM', 'SPHINCS', 'FALCON']
        power_consumption = [2.1, 2.3, 2.5, 2.7, 6.2, 6.5, 6.8, 7.1]
        security_levels = [3, 3, 4, 4, 8, 8, 9, 8]  # Relative security scale 1-10
        algorithm_types = ['Pre-Quantum']*4 + ['Post-Quantum']*4
        
        # 1. Power Consumption Comparison
        ax1.set_title('Power Consumption by Algorithm', fontweight='bold')
        colors = ['lightblue']*4 + ['lightcoral']*4
        bars1 = ax1.bar(algorithms, power_consumption, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_ylabel('Power Consumption (Watts)')
        ax1.set_xlabel('Cryptographic Algorithm')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add power values on bars
        for bar, power in zip(bars1, power_consumption):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{power}W', ha='center', va='bottom', fontweight='bold')
        
        # Add efficiency threshold line
        ax1.axhline(y=4.0, color='red', linestyle='--', alpha=0.7, label='Efficiency Threshold')
        ax1.legend()
        
        # 2. Security Level Comparison
        ax2.set_title('Security Levels', fontweight='bold') 
        bars2 = ax2.bar(algorithms, security_levels, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_ylabel('Security Level (1-10 scale)')
        ax2.set_xlabel('Cryptographic Algorithm')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar, sec in zip(bars2, security_levels):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(sec), ha='center', va='bottom', fontweight='bold')
        
        # 3. Power vs Security Trade-off
        ax3.set_title('Power-Security Trade-off Analysis', fontweight='bold')
        
        # Scatter plot with different colors for pre/post quantum
        pre_quantum_mask = np.array(algorithm_types) == 'Pre-Quantum'
        
        ax3.scatter(np.array(power_consumption)[pre_quantum_mask], 
                   np.array(security_levels)[pre_quantum_mask],
                   c='lightblue', s=200, alpha=0.8, edgecolors='black',
                   label='Pre-Quantum', marker='o')
        
        ax3.scatter(np.array(power_consumption)[~pre_quantum_mask], 
                   np.array(security_levels)[~pre_quantum_mask],
                   c='lightcoral', s=200, alpha=0.8, edgecolors='black', 
                   label='Post-Quantum', marker='^')
        
        # Add algorithm names as labels
        for i, alg in enumerate(algorithms):
            ax3.annotate(alg, (power_consumption[i], security_levels[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold')
        
        ax3.set_xlabel('Power Consumption (Watts)')
        ax3.set_ylabel('Security Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Algorithm Selection Matrix
        ax4.set_title('Algorithm Recommendation Matrix', fontweight='bold')
        ax4.axis('off')
        
        # Create recommendation text
        recommendations = [
            "ALGORITHM SELECTION GUIDE",
            "=" * 40,
            "",
            "🔋 BATTERY CRITICAL (0-20%):",
            "  • Low Threat: ASCON (2.1W) - Ultra efficient",
            "  • Medium Threat: SPECK (2.3W) - Fast & light", 
            "  • High Threat: KYBER (6.2W) - Minimal PQ option",
            "",
            "⚡ BATTERY MODERATE (20-60%):",  
            "  • Low Threat: HIGHT (2.5W) - Balanced",
            "  • Medium Threat: CAMELLIA (2.7W) - Reliable",
            "  • High Threat: KYBER (6.2W) - PQ standard",
            "",
            "🔋 BATTERY FULL (60-100%):",
            "  • Low Threat: Any pre-quantum algorithm",
            "  • Medium Threat: DILITHIUM (6.5W) - Strong",
            "  • High Threat: FALCON (7.1W) - Maximum security",
            "",
            "🎯 MISSION CRITICAL:",
            "  • Always prefer higher security levels",
            "  • SPHINCS (6.8W) for maximum protection",
            "",
            "Key Trade-offs:",
            "• Pre-quantum: 2.1-2.7W, Security 3-4",
            "• Post-quantum: 6.2-7.1W, Security 8-9"
        ]
        
        y_pos = 0.98
        for line in recommendations:
            if line.startswith("ALGORITHM") or line.startswith("="):
                color = self.colors['primary']
                weight = 'bold' 
                size = 12
            elif line.startswith("🔋") or line.startswith("⚡") or line.startswith("🎯"):
                color = self.colors['accent']
                weight = 'bold'
                size = 10
            elif line.startswith("  •"):
                color = 'black'
                weight = 'normal'
                size = 9
            elif line.startswith("Key"):
                color = self.colors['success']
                weight = 'bold'
                size = 10
            else:
                color = 'black'
                weight = 'normal' 
                size = 9
                
            ax4.text(0.02, y_pos, line, transform=ax4.transAxes,
                    fontsize=size, color=color, fontweight=weight,
                    fontfamily='monospace')
            y_pos -= 0.04
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '03_action_space_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_reward_function_diagram(self):
        """Create detailed reward function visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Component Reward Function Architecture', fontsize=16, fontweight='bold')
        
        # 1. Reward Component Breakdown
        ax1.set_title('Reward Component Weights', fontweight='bold')
        
        components = ['Battery\nEfficiency', 'Security\nMatch', 'Expert\nAgreement']
        weights = [40, 40, 20]
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
        
        wedges, texts, autotexts = ax1.pie(weights, labels=components, colors=colors,
                                          autopct='%1.1f%%', startangle=90, 
                                          explode=(0.05, 0.05, 0.05))
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        # 2. Reward Function Heatmap
        ax2.set_title('Reward Matrix: State-Action Examples', fontweight='bold')
        
        # Create sample reward matrix for visualization
        states_sample = ['Crit+Low', 'Crit+High', 'Med+Med', 'Full+Low', 'Full+High']
        actions_sample = ['ASCON', 'SPECK', 'KYBER', 'FALCON']
        
        # Sample reward values (realistic examples)
        reward_matrix = np.array([
            [8.5, 7.2, 2.1, -1.5],  # Critical+Low
            [-2.1, -1.8, 8.4, 7.9], # Critical+High  
            [6.2, 5.8, 7.1, 4.2],   # Medium+Medium
            [9.1, 8.7, 4.5, 2.8],   # Full+Low
            [5.2, 4.8, 8.8, 9.2]    # Full+High
        ])
        
        im = ax2.imshow(reward_matrix, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=10)
        ax2.set_xticks(range(len(actions_sample)))
        ax2.set_xticklabels(actions_sample)
        ax2.set_yticks(range(len(states_sample)))
        ax2.set_yticklabels(states_sample)
        ax2.set_xlabel('Actions (Algorithms)')
        ax2.set_ylabel('States (Battery+Threat)')
        
        # Add reward values as text
        for i in range(len(states_sample)):
            for j in range(len(actions_sample)):
                text_color = 'white' if reward_matrix[i,j] < 3 else 'black'
                ax2.text(j, i, f'{reward_matrix[i,j]:.1f}',
                        ha='center', va='center', color=text_color, fontweight='bold')
        
        plt.colorbar(im, ax=ax2, label='Reward Value')
        
        # 3. Component Calculation Example
        ax3.set_title('Reward Calculation Breakdown', fontweight='bold')
        ax3.axis('off')
        
        # Detailed calculation example
        calc_text = [
            "EXAMPLE: Critical Battery + High Threat → KYBER",
            "=" * 45,
            "",
            "💾 Battery Efficiency Component (40% weight):",
            f"  Power: KYBER = 6.2W vs Critical Battery",
            f"  Efficiency Score: {6.0:.1f} (good for security needs)",
            f"  Weighted: 0.4 × {6.0:.1f} = {0.4*6.0:.1f}",
            "",
            "🔒 Security Appropriateness (40% weight):",
            f"  High Threat + Post-Quantum = Perfect Match",
            f"  Security Score: {10.0:.1f} (maximum)",
            f"  Weighted: 0.4 × {10.0:.1f} = {0.4*10.0:.1f}",
            "",
            "👨‍🎓 Expert Agreement (20% weight):",
            f"  Expert Choice: KYBER ✓",
            f"  Agreement Score: {10.0:.1f} (perfect match)",
            f"  Weighted: 0.2 × {10.0:.1f} = {0.2*10.0:.1f}",
            "",
            f"🎯 TOTAL REWARD:",
            f"  {0.4*6.0:.1f} + {0.4*10.0:.1f} + {0.2*10.0:.1f} = {0.4*6.0 + 0.4*10.0 + 0.2*10.0:.1f}",
            "",
            "Grade: A (Excellent Choice!)"
        ]
        
        y_text = 0.98
        for line in calc_text:
            if line.startswith("EXAMPLE") or line.startswith("="):
                color = self.colors['primary']
                weight = 'bold'
                size = 11
            elif line.startswith("💾") or line.startswith("🔒") or line.startswith("👨‍🎓"):
                color = self.colors['secondary']
                weight = 'bold'
                size = 10
            elif line.startswith("🎯"):
                color = self.colors['success']
                weight = 'bold'
                size = 11
            elif "Grade:" in line:
                color = self.colors['accent']
                weight = 'bold'
                size = 11
            else:
                color = 'black'
                weight = 'normal'
                size = 9
            
            ax3.text(0.02, y_text, line, transform=ax3.transAxes,
                    fontsize=size, color=color, fontweight=weight,
                    fontfamily='monospace')
            y_text -= 0.045
        
        # 4. Reward Distribution Analysis
        ax4.set_title('Reward Score Distribution', fontweight='bold')
        
        # Generate sample reward distributions
        excellent_rewards = np.random.normal(8.5, 1.0, 1000)
        good_rewards = np.random.normal(5.5, 1.5, 800) 
        poor_rewards = np.random.normal(1.0, 2.0, 600)
        bad_rewards = np.random.normal(-2.0, 1.0, 400)
        
        ax4.hist(excellent_rewards, bins=30, alpha=0.7, color=self.colors['success'], 
                label='Excellent (8-10)', density=True)
        ax4.hist(good_rewards, bins=30, alpha=0.7, color=self.colors['info'],
                label='Good (4-7)', density=True)
        ax4.hist(poor_rewards, bins=30, alpha=0.7, color=self.colors['accent'],
                label='Poor (0-3)', density=True) 
        ax4.hist(bad_rewards, bins=30, alpha=0.7, color='red',
                label='Bad (<0)', density=True)
        
        ax4.set_xlabel('Reward Score')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add performance thresholds
        ax4.axvline(x=7, color='green', linestyle='--', alpha=0.8, label='Target Performance')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Minimum Acceptable')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '04_reward_function_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_learning_pipeline_diagram(self):
        """Create learning pipeline and algorithm comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Learning Pipeline & Algorithm Comparison', fontsize=16, fontweight='bold')
        
        # 1. Training Pipeline Flowchart
        ax1.set_title('Training Pipeline Process', fontweight='bold')
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.axis('off')
        
        # Pipeline steps
        steps = [
            ('Environment\nSetup', 2, 8.5, 1.8, 0.8),
            ('Agent\nInitialization', 6, 8.5, 1.8, 0.8),
            ('Episode\nLoop', 2, 6.5, 1.8, 0.8), 
            ('Action\nSelection', 6, 6.5, 1.8, 0.8),
            ('Environment\nStep', 2, 4.5, 1.8, 0.8),
            ('Reward\nCalculation', 6, 4.5, 1.8, 0.8),
            ('Learning\nUpdate', 2, 2.5, 1.8, 0.8),
            ('Performance\nEvaluation', 6, 2.5, 1.8, 0.8)
        ]
        
        colors_pipeline = [self.colors['primary'], self.colors['secondary']] * 4
        
        for i, (text, x, y, w, h) in enumerate(steps):
            box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle="round,pad=0.1",
                               facecolor=colors_pipeline[i],
                               edgecolor='black', linewidth=2)
            ax1.add_patch(box)
            ax1.text(x, y, text, ha='center', va='center',
                    fontweight='bold', fontsize=9, color='white')
        
        # Add arrows
        arrow_coords = [
            ((2.9, 8.5), (5.1, 8.5)),  # Setup → Init
            ((6, 7.7), (2, 7.3)),      # Init → Episode (curved)
            ((2.9, 6.5), (5.1, 6.5)),  # Episode → Action
            ((6, 5.7), (2, 5.3)),      # Action → Env Step
            ((2.9, 4.5), (5.1, 4.5)),  # Env → Reward
            ((6, 3.7), (2, 3.3)),      # Reward → Learning
            ((2.9, 2.5), (5.1, 2.5)),  # Learning → Eval
        ]
        
        for (x1, y1), (x2, y2) in arrow_coords:
            ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # 2. Q-Learning vs DQN Architecture
        ax2.set_title('Algorithm Architecture Comparison', fontweight='bold')
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        
        # Q-Learning side
        ax2.text(2.5, 9, 'Q-Learning (Tabular)', fontsize=14, fontweight='bold',
                ha='center', color=self.colors['primary'])
        
        q_components = [
            ('Q-Table\n30×8', 2.5, 7.5, 2, 0.8),
            ('ε-Greedy\nSelection', 2.5, 6, 2, 0.8),
            ('Q-Update\nRule', 2.5, 4.5, 2, 0.8),
            ('Convergence\nGuaranteed', 2.5, 3, 2, 0.8)
        ]
        
        for text, x, y, w, h in q_components:
            box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle="round,pad=0.05",
                               facecolor=self.colors['primary'],
                               alpha=0.3, edgecolor=self.colors['primary'])
            ax2.add_patch(box)
            ax2.text(x, y, text, ha='center', va='center', fontsize=9)
        
        # DQN side
        ax2.text(7.5, 9, 'Deep Q-Network', fontsize=14, fontweight='bold',
                ha='center', color=self.colors['secondary'])
        
        dqn_components = [
            ('Neural Network\n7→64→32→16→8', 7.5, 7.5, 2, 0.8),
            ('Experience\nReplay', 7.5, 6, 2, 0.8),
            ('Target Network\nStabilization', 7.5, 4.5, 2, 0.8),
            ('Generalization\nCapability', 7.5, 3, 2, 0.8)
        ]
        
        for text, x, y, w, h in dqn_components:
            box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle="round,pad=0.05", 
                               facecolor=self.colors['secondary'],
                               alpha=0.3, edgecolor=self.colors['secondary'])
            ax2.add_patch(box)
            ax2.text(x, y, text, ha='center', va='center', fontsize=9)
        
        # Add comparison arrows
        ax2.annotate('vs', xy=(5, 6), xytext=(5, 6),
                    fontsize=20, fontweight='bold', ha='center', va='center')
        
        # 3. Training Performance Simulation
        ax3.set_title('Expected Training Performance', fontweight='bold')
        
        # Simulate training curves
        episodes = np.linspace(0, 1000, 1000)
        
        # Q-Learning curve (faster initial learning, plateaus)
        q_learning_rewards = 50 * (1 - np.exp(-episodes/200)) - 20 + np.random.normal(0, 2, 1000)
        
        # DQN curve (slower start, higher final performance)
        dqn_rewards = 60 * (1 - np.exp(-episodes/300)) - 25 + np.random.normal(0, 3, 1000)
        
        # Smooth the curves
        if HAS_SCIPY:
            q_smooth = gaussian_filter1d(q_learning_rewards, sigma=20)
            dqn_smooth = gaussian_filter1d(dqn_rewards, sigma=20)
        else:
            # Simple moving average fallback
            window = 40
            q_smooth = np.convolve(q_learning_rewards, np.ones(window)/window, mode='same')
            dqn_smooth = np.convolve(dqn_rewards, np.ones(window)/window, mode='same')
        
        ax3.plot(episodes, q_smooth, color=self.colors['primary'], linewidth=3,
                label='Q-Learning', alpha=0.8)
        ax3.plot(episodes, dqn_smooth, color=self.colors['secondary'], linewidth=3,
                label='DQN', alpha=0.8)
        
        # Add confidence intervals
        ax3.fill_between(episodes, q_smooth-5, q_smooth+5, 
                        color=self.colors['primary'], alpha=0.2)
        ax3.fill_between(episodes, dqn_smooth-7, dqn_smooth+7,
                        color=self.colors['secondary'], alpha=0.2)
        
        ax3.set_xlabel('Training Episodes')
        ax3.set_ylabel('Average Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-30, 50)
        
        # Add performance milestones
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Break-even')
        ax3.axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Target Performance')
        
        # 4. Performance Metrics Comparison
        ax4.set_title('Algorithm Performance Comparison', fontweight='bold')
        
        metrics = ['Convergence\nSpeed', 'Final\nPerformance', 'Memory\nUsage', 
                  'Interpretability', 'Scalability']
        q_scores = [8, 6, 9, 10, 4]
        dqn_scores = [5, 9, 6, 4, 9]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, q_scores, width, label='Q-Learning',
                       color=self.colors['primary'], alpha=0.8)
        bars2 = ax4.bar(x + width/2, dqn_scores, width, label='DQN', 
                       color=self.colors['secondary'], alpha=0.8)
        
        ax4.set_xlabel('Performance Metrics')
        ax4.set_ylabel('Score (1-10)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45, ha='right')
        ax4.legend()
        ax4.set_ylim(0, 11)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '05_learning_pipeline.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Generate all architecture documentation visuals"""
    visualizer = ArchitectureVisualizer()
    visualizer.generate_all_diagrams()

if __name__ == "__main__":
    main()
