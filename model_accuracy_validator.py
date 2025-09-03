#!/usr/bin/env python3
"""
Detailed Model Accuracy Validation
=================================

This script tests both models against expert recommendations for all 30 states
to provide precise accuracy metrics for the professor presentation.

Author: RL Team
Date: September 4, 2025
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment.crypto_environment import CryptoEnvironment
from src.environment.state_space import StateSpace, CryptoState, BatteryLevel, ThreatStatus, MissionCriticality, CryptoAlgorithm
from src.algorithms.q_learning import QLearningAgent
from src.algorithms.deep_q_learning import DQNAgent

class ModelAccuracyValidator:
    """Validate model accuracy against expert recommendations"""
    
    def __init__(self):
        print("🎯 Model Accuracy Validator Initialized")
        self.env = CryptoEnvironment()
        self.state_space = StateSpace()
        
        # Generate all 30 states systematically
        self.all_states = []
        self.state_descriptions = []
        
        for battery in range(5):  # CRITICAL to HIGH
            for threat in range(3):  # NORMAL to CONFIRMED  
                for mission in range(2):  # ROUTINE to IMPORTANT
                    state = CryptoState(
                        battery_level=BatteryLevel(battery),
                        threat_status=ThreatStatus(threat),
                        mission_criticality=MissionCriticality(mission)
                    )
                    self.all_states.append(state)
                    
                    # Create description
                    battery_name = ["CRITICAL", "LOW", "MEDIUM", "GOOD", "HIGH"][battery]
                    threat_name = ["NORMAL", "CONFIRMING", "CONFIRMED"][threat]
                    mission_name = ["ROUTINE", "IMPORTANT"][mission]
                    desc = f"{battery_name}+{threat_name}+{mission_name}"
                    self.state_descriptions.append(desc)
        
        print(f"✅ Generated all {len(self.all_states)} states for validation")

    def initialize_models(self):
        """Initialize both models with expert knowledge"""
        print("\n🤖 Initializing models for accuracy testing...")
        
        # Initialize Q-Learning Agent
        print("   Initializing Q-Learning Agent...")
        self.q_agent = QLearningAgent(
            state_space=self.state_space,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon_start=0.01,  # Minimal exploration for testing
            warm_start=True
        )
        print("✅ Q-Learning Agent initialized")
        
        # Initialize DQN Agent  
        print("   Initializing DQN Agent...")
        self.dqn_agent = DQNAgent(
            state_space=self.state_space,
            learning_rate=0.001,
            batch_size=32,
            buffer_capacity=100000,
            target_update_frequency=100,
            warm_start=True
        )
        print("✅ DQN Agent initialized")

    def test_model_accuracy(self, model_name, agent):
        """Test a specific model's accuracy against expert recommendations"""
        print(f"\n🧠 Testing {model_name} Accuracy...")
        
        correct_predictions = 0
        total_predictions = 0
        detailed_results = []
        
        for i, (state, description) in enumerate(zip(self.all_states, self.state_descriptions)):
            try:
                # Get expert recommendation
                expert_action = self.state_space.get_expert_action(state)
                expert_action_idx = expert_action.value
                
                # Get model prediction
                if model_name == "Q-Learning":
                    state_idx = state.to_index()
                    model_action_idx = agent.select_action(state_idx, training=False)
                else:  # DQN
                    state_vector = self.state_space.encode_state_for_dqn(i)
                    model_action_idx = agent.select_action(state_vector, training=False)
                
                # Check if prediction matches expert
                is_correct = (model_action_idx == expert_action_idx)
                if is_correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Store detailed result
                result = {
                    'state_id': i,
                    'state_description': description,
                    'expert_action': expert_action.name,
                    'expert_action_idx': expert_action_idx,
                    'model_action_idx': model_action_idx,
                    'model_action': CryptoAlgorithm(model_action_idx).name,
                    'is_correct': is_correct,
                    'battery_level': state.battery_level.name,
                    'threat_status': state.threat_status.name,
                    'mission_criticality': state.mission_criticality.name
                }
                detailed_results.append(result)
                
                # Print progress every 10 states
                if (i + 1) % 10 == 0:
                    print(f"   Tested {i + 1}/30 states...")
                
            except Exception as e:
                print(f"   ❌ Error testing state {i}: {str(e)}")
                detailed_results.append({
                    'state_id': i,
                    'state_description': description,
                    'error': str(e),
                    'is_correct': False
                })
                total_predictions += 1
        
        # Calculate accuracy
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        print(f"✅ {model_name} Testing Complete:")
        print(f"   📊 Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions} correct)")
        
        return accuracy, correct_predictions, total_predictions, detailed_results

    def create_accuracy_report(self, q_results, dqn_results):
        """Create comprehensive accuracy report"""
        q_accuracy, q_correct, q_total, q_details = q_results
        dqn_accuracy, dqn_correct, dqn_total, dqn_details = dqn_results
        
        print("\n📊 Creating Accuracy Report...")
        
        # Create summary statistics
        report_data = {
            "validation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_states_tested": 30,
            "q_learning": {
                "accuracy_percentage": q_accuracy,
                "correct_predictions": q_correct,
                "total_predictions": q_total,
                "error_count": q_total - q_correct
            },
            "dqn": {
                "accuracy_percentage": dqn_accuracy,
                "correct_predictions": dqn_correct,
                "total_predictions": dqn_total,
                "error_count": dqn_total - dqn_correct
            }
        }
        
        # Analyze error patterns
        q_errors = [r for r in q_details if not r.get('is_correct', False)]
        dqn_errors = [r for r in dqn_details if not r.get('is_correct', False)]
        
        # Battery level error analysis
        battery_error_analysis = {
            "q_learning": {},
            "dqn": {}
        }
        
        for level in ["CRITICAL", "LOW", "MEDIUM", "GOOD", "HIGH"]:
            q_level_errors = [e for e in q_errors if e.get('battery_level') == level]
            dqn_level_errors = [e for e in dqn_errors if e.get('battery_level') == level]
            
            battery_error_analysis["q_learning"][level] = len(q_level_errors)
            battery_error_analysis["dqn"][level] = len(dqn_level_errors)
        
        report_data["error_analysis"] = battery_error_analysis
        
        # Print detailed report
        print("\n" + "="*60)
        print("🎯 DETAILED ACCURACY VALIDATION REPORT")
        print("="*60)
        
        print(f"\n📊 OVERALL ACCURACY:")
        print(f"   Q-Learning: {q_accuracy:.1f}% ({q_correct}/{q_total} correct)")
        print(f"   DQN:        {dqn_accuracy:.1f}% ({dqn_correct}/{dqn_total} correct)")
        
        if q_accuracy > dqn_accuracy:
            print(f"   🏆 Winner: Q-Learning (+{q_accuracy - dqn_accuracy:.1f}%)")
        elif dqn_accuracy > q_accuracy:
            print(f"   🏆 Winner: DQN (+{dqn_accuracy - q_accuracy:.1f}%)")
        else:
            print(f"   ⚖️ Tie: Both models equal accuracy")
        
        # Show error breakdown by battery level
        print(f"\n🔋 ERROR BREAKDOWN BY BATTERY LEVEL:")
        for level in ["CRITICAL", "LOW", "MEDIUM", "GOOD", "HIGH"]:
            q_errs = battery_error_analysis["q_learning"][level]
            dqn_errs = battery_error_analysis["dqn"][level]
            print(f"   {level:8}: Q-Learning={q_errs} errors, DQN={dqn_errs} errors")
        
        # Show some example correct predictions
        q_correct_examples = [r for r in q_details if r.get('is_correct', False)][:5]
        print(f"\n✅ Q-LEARNING CORRECT PREDICTIONS (Sample):")
        for example in q_correct_examples:
            print(f"   State {example['state_id']:2}: {example['state_description']:25} → {example['expert_action']:8} ✓")
        
        # Show some example errors (if any)
        if q_errors:
            print(f"\n❌ Q-LEARNING INCORRECT PREDICTIONS:")
            for error in q_errors[:3]:  # Show first 3 errors
                expert = error.get('expert_action', 'N/A')
                model = error.get('model_action', 'N/A')
                print(f"   State {error['state_id']:2}: {error['state_description']:25} → Expert: {expert:8}, Model: {model:8} ✗")
        
        return report_data

    def run_comprehensive_accuracy_test(self):
        """Run complete accuracy validation"""
        print("\n" + "="*60)
        print("🎯 STARTING COMPREHENSIVE ACCURACY VALIDATION")
        print("="*60)
        
        # Initialize models
        self.initialize_models()
        
        # Test Q-Learning accuracy
        q_results = self.test_model_accuracy("Q-Learning", self.q_agent)
        
        # Test DQN accuracy
        dqn_results = self.test_model_accuracy("Deep Q-Network", self.dqn_agent)
        
        # Create comprehensive report
        report_data = self.create_accuracy_report(q_results, dqn_results)
        
        # Save results
        import json
        with open('outputs/validation_results/detailed_accuracy_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed accuracy report saved to: outputs/validation_results/detailed_accuracy_report.json")
        
        return report_data

def main():
    """Main execution function"""
    print("🎯 Model Accuracy Validation System")
    print("=" * 50)
    
    try:
        validator = ModelAccuracyValidator()
        results = validator.run_comprehensive_accuracy_test()
        
        print("\n" + "="*60)
        print("✅ ACCURACY VALIDATION COMPLETE!")
        print("="*60)
        print(f"🎯 Ready for Professor Presentation with Precise Accuracy Metrics!")
        
    except Exception as e:
        print(f"\n❌ Error during validation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
