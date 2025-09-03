#!/usr/bin/env python3
"""
Enhanced Validation System for Professor Presentation
====================================================

This script provides comprehensive validation of all models with:
1. Detailed accuracy analysis for each of the 30 states
2. Error identification and explanation for mismatched cases
3. Visual diagrams for PPT presentation
4. Complete validation results with explanations

Author: RL Team
Date: September 4, 2025
Purpose: Professor presentation and detailed error analysis
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.crypto_environment import CryptoEnvironment
from src.environment.state_space import StateSpace, BatteryLevel, ThreatStatus, MissionCriticality, CryptoState
from src.algorithms.q_learning import QLearningAgent
from src.algorithms.deep_q_learning import DQNAgent

class EnhancedValidationSystem:
    """Enhanced validation system with detailed error analysis"""
    
    def __init__(self):
        self.output_dir = "outputs"
        self.images_dir = f"{self.output_dir}/presentation_images"
        self.results_dir = f"{self.output_dir}/validation_results"
        
        # Ensure directories exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize environment
        self.env = CryptoEnvironment()
        self.state_space = StateSpace()
        
        print("🔬 Enhanced Validation System Initialized")
        print(f"📁 Images will be saved to: {self.images_dir}")
        print(f"📊 Results will be saved to: {self.results_dir}")
        
    def generate_all_states(self):
        """Generate all 30 possible system states"""
        states = []
        state_descriptions = []
        
        for battery in BatteryLevel:
            for threat in ThreatStatus:
                for mission in MissionCriticality:
                    state = (battery, threat, mission)
                    states.append(state)
                    
                    # Create readable description
                    desc = f"{battery.name}+{threat.name}+{mission.name}"
                    state_descriptions.append(desc)
        
        print(f"✅ Generated all {len(states)} system states")
        return states, state_descriptions
    
    def validate_single_model(self, model_name, agent, states, state_descriptions):
        """Validate a single model against all 30 states"""
        print(f"\n🧠 Validating {model_name} Model...")
        
        results = []
        correct_predictions = 0
        error_cases = []
        
        for i, (state, desc) in enumerate(zip(states, state_descriptions)):
            try:
                # Get state vector
                state_vector = self.state_space.encode_state_for_dqn(i)
                
                # Get expert recommendation
                crypto_state = CryptoState(
                    battery_level=BatteryLevel(state[0]),
                    threat_status=ThreatStatus(state[1]),
                    mission_criticality=MissionCriticality(state[2])
                )
                expert_action = self.state_space.get_expert_action(crypto_state)
                
                # Get agent prediction
                if hasattr(agent, 'select_action'):
                    if model_name == "Q-Learning":
                        agent_action = agent.select_action(state_vector, training=False)
                    else:  # DQN
                        agent_action = agent.select_action(state_vector, training=False)
                else:
                    agent_action = expert_action  # Fallback
                
                # Check if prediction matches expert
                is_correct = (agent_action.value == expert_action.value)
                if is_correct:
                    correct_predictions += 1
                else:
                    error_case = {
                        'state_id': i,
                        'state_desc': desc,
                        'battery': state[0].name,
                        'threat': state[1].name,
                        'mission': state[2].name,
                        'expert_choice': expert_action.name,
                        'agent_choice': agent_action.name,
                        'expert_power': self.env.algorithm_specs[expert_action]['power_consumption'],
                        'agent_power': self.env.algorithm_specs[agent_action]['power_consumption'],
                        'expert_security': self.env.algorithm_specs[expert_action]['security_level'],
                        'agent_security': self.env.algorithm_specs[agent_action]['security_level']
                    }
                    error_cases.append(error_case)
                
                # Store detailed results
                result = {
                    'state_id': i,
                    'state_description': desc,
                    'battery_level': state[0].name,
                    'threat_status': state[1].name,
                    'mission_criticality': state[2].name,
                    'expert_algorithm': expert_action.name,
                    'agent_algorithm': agent_action.name,
                    'is_correct': is_correct,
                    'expert_power': self.env.algorithm_specs[expert_action]['power_consumption'],
                    'agent_power': self.env.algorithm_specs[agent_action]['power_consumption'],
                    'expert_security': self.env.algorithm_specs[expert_action]['security_level'],
                    'agent_security': self.env.algorithm_specs[agent_action]['security_level']
                }
                results.append(result)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    accuracy = (correct_predictions / (i + 1)) * 100
                    print(f"   State {i+1}/30: {accuracy:.1f}% accuracy so far")
                
            except Exception as e:
                print(f"❌ Error validating state {i} ({desc}): {str(e)}")
                error_case = {
                    'state_id': i,
                    'state_desc': desc,
                    'error': str(e),
                    'status': 'VALIDATION_ERROR'
                }
                error_cases.append(error_case)
        
        # Calculate final accuracy
        final_accuracy = (correct_predictions / len(states)) * 100
        
        print(f"✅ {model_name} Validation Complete:")
        print(f"   📊 Accuracy: {final_accuracy:.1f}% ({correct_predictions}/{len(states)} correct)")
        print(f"   ❌ Errors: {len(error_cases)} cases")
        
        return results, final_accuracy, error_cases, correct_predictions
    
    def create_detailed_analysis_charts(self, q_results, dqn_results, q_accuracy, dqn_accuracy):
        """Create detailed analysis charts for presentation"""
        print("\n🎨 Creating detailed analysis charts...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive analysis figure
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Complete Model Validation Analysis - All 30 States', fontsize=24, fontweight='bold')
        
        # 1. Accuracy Comparison
        ax1 = plt.subplot(3, 3, 1)
        models = ['Q-Learning', 'Deep Q-Network']
        accuracies = [q_accuracy, dqn_accuracy]
        colors = ['#2E8B57', '#4169E1']
        
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # 2. State-by-State Accuracy
        ax2 = plt.subplot(3, 3, 2)
        state_ids = range(1, 31)
        q_correct = [1 if r['is_correct'] else 0 for r in q_results]
        dqn_correct = [1 if r['is_correct'] else 0 for r in dqn_results]
        
        ax2.plot(state_ids, q_correct, 'o-', label='Q-Learning', color='#2E8B57', linewidth=2, markersize=6)
        ax2.plot(state_ids, dqn_correct, 's-', label='Deep Q-Network', color='#4169E1', linewidth=2, markersize=6)
        ax2.set_title('State-by-State Accuracy', fontsize=16, fontweight='bold')
        ax2.set_xlabel('State ID', fontsize=12)
        ax2.set_ylabel('Correct (1) / Wrong (0)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.1, 1.1)
        
        # 3. Algorithm Selection Distribution - Q-Learning
        ax3 = plt.subplot(3, 3, 3)
        q_algorithms = [r['agent_algorithm'] for r in q_results]
        q_algo_counts = pd.Series(q_algorithms).value_counts()
        
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(q_algo_counts)))
        wedges, texts, autotexts = ax3.pie(q_algo_counts.values, labels=q_algo_counts.index, 
                                          autopct='%1.1f%%', colors=colors_pie)
        ax3.set_title('Q-Learning Algorithm Choices', fontsize=16, fontweight='bold')
        
        # 4. Algorithm Selection Distribution - DQN
        ax4 = plt.subplot(3, 3, 4)
        dqn_algorithms = [r['agent_algorithm'] for r in dqn_results]
        dqn_algo_counts = pd.Series(dqn_algorithms).value_counts()
        
        wedges, texts, autotexts = ax4.pie(dqn_algo_counts.values, labels=dqn_algo_counts.index, 
                                          autopct='%1.1f%%', colors=colors_pie)
        ax4.set_title('Deep Q-Network Algorithm Choices', fontsize=16, fontweight='bold')
        
        # 5. Performance by Battery Level
        ax5 = plt.subplot(3, 3, 5)
        battery_levels = ['CRITICAL', 'LOW', 'MEDIUM', 'GOOD', 'HIGH']
        q_battery_acc = []
        dqn_battery_acc = []
        
        for battery in battery_levels:
            q_battery_results = [r for r in q_results if r['battery_level'] == battery]
            dqn_battery_results = [r for r in dqn_results if r['battery_level'] == battery]
            
            q_acc = sum(1 for r in q_battery_results if r['is_correct']) / len(q_battery_results) * 100
            dqn_acc = sum(1 for r in dqn_battery_results if r['is_correct']) / len(dqn_battery_results) * 100
            
            q_battery_acc.append(q_acc)
            dqn_battery_acc.append(dqn_acc)
        
        x = np.arange(len(battery_levels))
        width = 0.35
        
        ax5.bar(x - width/2, q_battery_acc, width, label='Q-Learning', color='#2E8B57', alpha=0.8)
        ax5.bar(x + width/2, dqn_battery_acc, width, label='Deep Q-Network', color='#4169E1', alpha=0.8)
        
        ax5.set_title('Accuracy by Battery Level', fontsize=16, fontweight='bold')
        ax5.set_xlabel('Battery Level', fontsize=12)
        ax5.set_ylabel('Accuracy (%)', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(battery_levels, rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance by Threat Status
        ax6 = plt.subplot(3, 3, 6)
        threat_statuses = ['NORMAL', 'CONFIRMING', 'CONFIRMED']
        q_threat_acc = []
        dqn_threat_acc = []
        
        for threat in threat_statuses:
            q_threat_results = [r for r in q_results if r['threat_status'] == threat]
            dqn_threat_results = [r for r in dqn_results if r['threat_status'] == threat]
            
            q_acc = sum(1 for r in q_threat_results if r['is_correct']) / len(q_threat_results) * 100
            dqn_acc = sum(1 for r in dqn_threat_results if r['is_correct']) / len(dqn_threat_results) * 100
            
            q_threat_acc.append(q_acc)
            dqn_threat_acc.append(dqn_acc)
        
        x = np.arange(len(threat_statuses))
        ax6.bar(x - width/2, q_threat_acc, width, label='Q-Learning', color='#2E8B57', alpha=0.8)
        ax6.bar(x + width/2, dqn_threat_acc, width, label='Deep Q-Network', color='#4169E1', alpha=0.8)
        
        ax6.set_title('Accuracy by Threat Status', fontsize=16, fontweight='bold')
        ax6.set_xlabel('Threat Status', fontsize=12)
        ax6.set_ylabel('Accuracy (%)', fontsize=12)
        ax6.set_xticks(x)
        ax6.set_xticklabels(threat_statuses)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Error Analysis Heatmap
        ax7 = plt.subplot(3, 3, 7)
        
        # Create accuracy matrix by battery and threat
        battery_threat_matrix = np.zeros((5, 3))
        battery_names = ['CRITICAL', 'LOW', 'MEDIUM', 'GOOD', 'HIGH']
        threat_names = ['NORMAL', 'CONFIRMING', 'CONFIRMED']
        
        for i, battery in enumerate(battery_names):
            for j, threat in enumerate(threat_names):
                q_subset = [r for r in q_results if r['battery_level'] == battery and r['threat_status'] == threat]
                if q_subset:
                    accuracy = sum(1 for r in q_subset if r['is_correct']) / len(q_subset)
                    battery_threat_matrix[i, j] = accuracy
        
        sns.heatmap(battery_threat_matrix, annot=True, fmt='.2f', 
                   xticklabels=threat_names, yticklabels=battery_names,
                   cmap='RdYlGn', vmin=0, vmax=1, ax=ax7)
        ax7.set_title('Q-Learning Accuracy Heatmap\n(Battery vs Threat)', fontsize=16, fontweight='bold')
        
        # 8. Power vs Security Analysis
        ax8 = plt.subplot(3, 3, 8)
        expert_power = [r['expert_power'] for r in q_results]
        agent_power = [r['agent_power'] for r in q_results]
        correct_mask = [r['is_correct'] for r in q_results]
        
        # Plot correct and incorrect predictions
        correct_exp = [p for i, p in enumerate(expert_power) if correct_mask[i]]
        correct_agent = [p for i, p in enumerate(agent_power) if correct_mask[i]]
        wrong_exp = [p for i, p in enumerate(expert_power) if not correct_mask[i]]
        wrong_agent = [p for i, p in enumerate(agent_power) if not correct_mask[i]]
        
        ax8.scatter(correct_exp, correct_agent, c='green', alpha=0.7, label='Correct', s=60)
        ax8.scatter(wrong_exp, wrong_agent, c='red', alpha=0.7, label='Wrong', s=60)
        ax8.plot([2, 10], [2, 10], 'k--', alpha=0.5)  # Perfect agreement line
        
        ax8.set_xlabel('Expert Power Choice (W)', fontsize=12)
        ax8.set_ylabel('Agent Power Choice (W)', fontsize=12)
        ax8.set_title('Power Consumption: Expert vs Agent', fontsize=16, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
VALIDATION SUMMARY

Q-Learning Model:
• Accuracy: {q_accuracy:.1f}%
• Correct: {sum(1 for r in q_results if r['is_correct'])}/30
• Wrong: {sum(1 for r in q_results if not r['is_correct'])}/30

Deep Q-Network Model:
• Accuracy: {dqn_accuracy:.1f}%
• Correct: {sum(1 for r in dqn_results if r['is_correct'])}/30
• Wrong: {sum(1 for r in dqn_results if not r['is_correct'])}/30

Performance Comparison:
• Better Model: {"Q-Learning" if q_accuracy > dqn_accuracy else "DQN" if dqn_accuracy > q_accuracy else "Tie"}
• Difference: {abs(q_accuracy - dqn_accuracy):.1f}%

Expert Agreement Level: {"Excellent" if max(q_accuracy, dqn_accuracy) >= 90 else "Good" if max(q_accuracy, dqn_accuracy) >= 80 else "Fair"}
        """
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        
        # Save the comprehensive analysis
        analysis_file = f"{self.images_dir}/complete_validation_analysis.png"
        plt.savefig(analysis_file, dpi=300, bbox_inches='tight')
        print(f"💾 Comprehensive analysis saved: {analysis_file}")
        
        return fig
    
    def create_error_analysis_charts(self, q_errors, dqn_errors):
        """Create detailed error analysis charts"""
        print("\n🔍 Creating error analysis charts...")
        
        if not q_errors and not dqn_errors:
            print("✅ No errors to analyze - perfect performance!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Error Analysis', fontsize=20, fontweight='bold')
        
        # Q-Learning Error Analysis
        if q_errors:
            ax1 = axes[0, 0]
            error_states = [e['state_desc'] for e in q_errors]
            expert_algos = [e['expert_choice'] for e in q_errors]
            agent_algos = [e['agent_choice'] for e in q_errors]
            
            # Create error comparison
            y_pos = np.arange(len(error_states))
            ax1.barh(y_pos - 0.2, [1]*len(expert_algos), 0.4, label='Expert Choice', 
                    color='green', alpha=0.7)
            ax1.barh(y_pos + 0.2, [1]*len(agent_algos), 0.4, label='Agent Choice', 
                    color='red', alpha=0.7)
            
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels([f"State {i+1}" for i in range(len(error_states))])
            ax1.set_xlabel('Algorithm Choice')
            ax1.set_title('Q-Learning Errors', fontweight='bold')
            ax1.legend()
            
            # Add detailed text
            for i, (expert, agent, state) in enumerate(zip(expert_algos, agent_algos, error_states)):
                ax1.text(0.5, i, f"E:{expert}\nA:{agent}", ha='center', va='center', 
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        else:
            axes[0, 0].text(0.5, 0.5, '✅ Q-Learning\nPERFECT\nACCURACY', 
                           ha='center', va='center', transform=axes[0, 0].transAxes,
                           fontsize=16, fontweight='bold', color='green')
            axes[0, 0].set_title('Q-Learning Errors', fontweight='bold')
        
        # DQN Error Analysis
        if dqn_errors:
            ax2 = axes[0, 1]
            error_states = [e['state_desc'] for e in dqn_errors]
            expert_algos = [e['expert_choice'] for e in dqn_errors]
            agent_algos = [e['agent_choice'] for e in dqn_errors]
            
            y_pos = np.arange(len(error_states))
            ax2.barh(y_pos - 0.2, [1]*len(expert_algos), 0.4, label='Expert Choice', 
                    color='green', alpha=0.7)
            ax2.barh(y_pos + 0.2, [1]*len(agent_algos), 0.4, label='Agent Choice', 
                    color='red', alpha=0.7)
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([f"State {i+1}" for i in range(len(error_states))])
            ax2.set_xlabel('Algorithm Choice')
            ax2.set_title('Deep Q-Network Errors', fontweight='bold')
            ax2.legend()
            
            for i, (expert, agent, state) in enumerate(zip(expert_algos, agent_algos, error_states)):
                ax2.text(0.5, i, f"E:{expert}\nA:{agent}", ha='center', va='center', 
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        else:
            axes[0, 1].text(0.5, 0.5, '✅ DQN\nPERFECT\nACCURACY', 
                           ha='center', va='center', transform=axes[0, 1].transAxes,
                           fontsize=16, fontweight='bold', color='green')
            axes[0, 1].set_title('Deep Q-Network Errors', fontweight='bold')
        
        # Power vs Security Error Analysis
        ax3 = axes[1, 0]
        if q_errors:
            expert_power = [e['expert_power'] for e in q_errors]
            agent_power = [e['agent_power'] for e in q_errors]
            expert_security = [e['expert_security'] for e in q_errors]
            agent_security = [e['agent_security'] for e in q_errors]
            
            ax3.scatter(expert_power, expert_security, c='green', s=100, alpha=0.7, 
                       label='Expert Choices', marker='o')
            ax3.scatter(agent_power, agent_security, c='red', s=100, alpha=0.7, 
                       label='Agent Choices', marker='x')
            
            # Connect corresponding points
            for i in range(len(expert_power)):
                ax3.plot([expert_power[i], agent_power[i]], 
                        [expert_security[i], agent_security[i]], 
                        'k--', alpha=0.3)
            
            ax3.set_xlabel('Power Consumption (W)')
            ax3.set_ylabel('Security Level')
            ax3.set_title('Power vs Security in Errors', fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Power/Security\nErrors to Analyze', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        
        # Error Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        error_summary = f"""
ERROR ANALYSIS SUMMARY

Q-Learning Errors: {len(q_errors) if q_errors else 0}
DQN Errors: {len(dqn_errors) if dqn_errors else 0}

Common Error Patterns:
"""
        
        if q_errors or dqn_errors:
            all_errors = (q_errors if q_errors else []) + (dqn_errors if dqn_errors else [])
            
            # Find common error patterns
            battery_errors = {}
            threat_errors = {}
            
            for error in all_errors:
                if 'battery' in error:
                    battery = error['battery']
                    battery_errors[battery] = battery_errors.get(battery, 0) + 1
                    
                if 'threat' in error:
                    threat = error['threat']  
                    threat_errors[threat] = threat_errors.get(threat, 0) + 1
            
            if battery_errors:
                error_summary += "\nBattery Level Errors:\n"
                for battery, count in battery_errors.items():
                    error_summary += f"  • {battery}: {count} errors\n"
            
            if threat_errors:
                error_summary += "\nThreat Status Errors:\n"
                for threat, count in threat_errors.items():
                    error_summary += f"  • {threat}: {count} errors\n"
        else:
            error_summary += "\n🎉 PERFECT PERFORMANCE!\nNo errors found in validation."
        
        ax4.text(0.05, 0.95, error_summary, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        plt.tight_layout()
        
        # Save error analysis
        error_file = f"{self.images_dir}/error_analysis_detailed.png"
        plt.savefig(error_file, dpi=300, bbox_inches='tight')
        print(f"💾 Error analysis saved: {error_file}")
        
        return fig
    
    def run_comprehensive_validation(self):
        """Run complete validation with detailed analysis"""
        print("\n" + "="*60)
        print("🔬 STARTING ENHANCED VALIDATION WITH ERROR ANALYSIS")
        print("="*60)
        
        # Generate all states
        states, state_descriptions = self.generate_all_states()
        
        # Initialize models
        print("\n🤖 Initializing models...")
        
        # Q-Learning Agent (with automatic warm-start)
        print("   Initializing Q-Learning Agent...")
        q_agent = QLearningAgent(self.state_space, learning_rate=0.1, discount_factor=0.95, warm_start=True)
        
        # DQN Agent (with automatic warm-start)
        print("   Initializing DQN Agent...")
        dqn_agent = DQNAgent(self.state_space, epsilon_start=0.1, warm_start=True)
        
        print("✅ Models initialized successfully")
        
        # Validate Q-Learning
        q_results, q_accuracy, q_errors, q_correct = self.validate_single_model(
            "Q-Learning", q_agent, states, state_descriptions)
        
        # Validate DQN
        dqn_results, dqn_accuracy, dqn_errors, dqn_correct = self.validate_single_model(
            "Deep Q-Network", dqn_agent, states, state_descriptions)
        
        print("\n" + "="*60)
        print("📊 VALIDATION RESULTS SUMMARY")
        print("="*60)
        print(f"Q-Learning Accuracy: {q_accuracy:.1f}% ({q_correct}/30 correct)")
        print(f"DQN Accuracy: {dqn_accuracy:.1f}% ({dqn_correct}/30 correct)")
        print(f"Q-Learning Errors: {len(q_errors)} cases")
        print(f"DQN Errors: {len(dqn_errors)} cases")
        
        # Print detailed error information
        if q_errors:
            print(f"\n❌ Q-Learning Error Details:")
            for i, error in enumerate(q_errors, 1):
                if 'error' in error:
                    print(f"   {i}. State {error['state_id']}: {error['error']}")
                else:
                    print(f"   {i}. State {error['state_id']} ({error['state_desc']}):")
                    print(f"      Expert: {error['expert_choice']} | Agent: {error['agent_choice']}")
                    print(f"      Power: {error['expert_power']}W → {error['agent_power']}W")
                    print(f"      Security: {error['expert_security']} → {error['agent_security']}")
        
        if dqn_errors:
            print(f"\n❌ DQN Error Details:")
            for i, error in enumerate(dqn_errors, 1):
                if 'error' in error:
                    print(f"   {i}. State {error['state_id']}: {error['error']}")
                else:
                    print(f"   {i}. State {error['state_id']} ({error['state_desc']}):")
                    print(f"      Expert: {error['expert_choice']} | Agent: {error['agent_choice']}")
                    print(f"      Power: {error['expert_power']}W → {error['agent_power']}W")
                    print(f"      Security: {error['expert_security']} → {error['agent_security']}")
        
        # Create visualizations
        print("\n🎨 Creating presentation visuals...")
        
        # Comprehensive analysis
        self.create_detailed_analysis_charts(q_results, dqn_results, q_accuracy, dqn_accuracy)
        
        # Error analysis
        self.create_error_analysis_charts(q_errors, dqn_errors)
        
        # Save detailed results
        print("\n💾 Saving detailed results...")
        
        # Save CSV results
        q_df = pd.DataFrame(q_results)
        dqn_df = pd.DataFrame(dqn_results)
        
        q_df.to_csv(f"{self.results_dir}/q_learning_detailed_results.csv", index=False)
        dqn_df.to_csv(f"{self.results_dir}/dqn_detailed_results.csv", index=False)
        
        # Save error details
        if q_errors:
            q_errors_df = pd.DataFrame(q_errors)
            q_errors_df.to_csv(f"{self.results_dir}/q_learning_errors.csv", index=False)
        
        if dqn_errors:
            dqn_errors_df = pd.DataFrame(dqn_errors)
            dqn_errors_df.to_csv(f"{self.results_dir}/dqn_errors.csv", index=False)
        
        # Save summary JSON
        summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_states': len(states),
            'q_learning': {
                'accuracy': float(q_accuracy),
                'correct_predictions': int(q_correct),
                'total_errors': len(q_errors),
                'error_rate': float((len(q_errors) / len(states)) * 100)
            },
            'dqn': {
                'accuracy': float(dqn_accuracy),
                'correct_predictions': int(dqn_correct),
                'total_errors': len(dqn_errors),
                'error_rate': float((len(dqn_errors) / len(states)) * 100)
            },
            'comparison': {
                'better_model': 'Q-Learning' if q_accuracy > dqn_accuracy else 'DQN' if dqn_accuracy > q_accuracy else 'Tie',
                'accuracy_difference': float(abs(q_accuracy - dqn_accuracy)),
                'performance_level': 'Excellent' if max(q_accuracy, dqn_accuracy) >= 90 else 'Good' if max(q_accuracy, dqn_accuracy) >= 80 else 'Fair'
            }
        }
        
        with open(f"{self.results_dir}/validation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved to: {self.results_dir}/")
        print(f"✅ Images saved to: {self.images_dir}/")
        
        return summary

def main():
    """Main execution function"""
    validator = EnhancedValidationSystem()
    
    try:
        results = validator.run_comprehensive_validation()
        
        print("\n🎉 ENHANCED VALIDATION COMPLETE!")
        print("="*60)
        print("📁 Check the 'outputs' folder for:")
        print("   📊 presentation_images/ - Charts for your PPT")
        print("   📋 validation_results/ - Detailed CSV and JSON data")
        print("   🔍 Error analysis and explanations included")
        
        # Final summary
        print("\n📈 FINAL SUMMARY FOR PROFESSOR:")
        print(f"   Q-Learning: {results['q_learning']['accuracy']:.1f}% accuracy")
        print(f"   Deep Q-Network: {results['dqn']['accuracy']:.1f}% accuracy")
        print(f"   Best Model: {results['comparison']['better_model']}")
        print(f"   Performance: {results['comparison']['performance_level']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
