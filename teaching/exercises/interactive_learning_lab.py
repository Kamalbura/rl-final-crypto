#!/usr/bin/env python3
"""
Interactive RL Learning Exercises for Team Training
Hands-on exercises to understand reinforcement learning concepts

Exercise Series:
1. State Space Explorer
2. Reward Function Calculator  
3. Q-Learning Visualizer
4. Algorithm Comparison Tool
5. Parameter Experiment Lab

Author: RL Team
Date: September 4, 2025
Purpose: Team education and hands-on learning
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Add src to path  
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from environment.state_space import StateSpace, BatteryLevel, ThreatLevel, MissionCriticality
from environment.crypto_environment import CryptoEnvironment

class InteractiveLearningLab:
    """Interactive exercises for learning RL concepts"""
    
    def __init__(self):
        self.state_space = StateSpace()
        self.env = CryptoEnvironment()
        print("🎓 Welcome to the Interactive RL Learning Lab!")
        print("📚 Designed for fast learners new to reinforcement learning")
        print("-" * 60)
    
    def run_all_exercises(self):
        """Run all exercises in sequence"""
        print("\n🚀 Starting Complete Learning Journey")
        print("="*50)
        
        exercises = [
            ("🗺️  Exercise 1: State Space Explorer", self.exercise_1_state_space),
            ("🎯 Exercise 2: Reward Calculator", self.exercise_2_rewards), 
            ("🧠 Exercise 3: Decision Simulator", self.exercise_3_decisions),
            ("📊 Exercise 4: Learning Visualizer", self.exercise_4_learning),
            ("🔬 Exercise 5: Algorithm Comparer", self.exercise_5_comparison)
        ]
        
        for name, exercise_func in exercises:
            print(f"\n{name}")
            print("-" * len(name))
            exercise_func()
            input("\n⏸️  Press Enter to continue to next exercise...")
        
        print("\n🎉 Congratulations! You've completed all RL learning exercises!")
    
    def exercise_1_state_space(self):
        """Exercise 1: Understanding the state space"""
        print("🎯 Goal: Understand how we represent different situations")
        print("\n📋 Our system has 30 different states:")
        print("   • 5 Battery Levels × 3 Threat Levels × 2 Mission Types = 30 states")
        
        # Show some example states
        print("\n🔍 Example States:")
        example_states = [
            (BatteryLevel.CRITICAL, ThreatLevel.HIGH, MissionCriticality.CRITICAL),
            (BatteryLevel.FULL, ThreatLevel.LOW, MissionCriticality.NORMAL),
            (BatteryLevel.MEDIUM, ThreatLevel.MEDIUM, MissionCriticality.CRITICAL)
        ]
        
        for i, (battery, threat, mission) in enumerate(example_states, 1):
            state_id = self.state_space.encode_state(battery, threat, mission)
            expert_action = self.state_space.get_expert_action(state_id)
            expert_name = self.state_space.get_action_name(expert_action)
            
            print(f"   {i}. Battery: {battery.name}, Threat: {threat.name}, Mission: {mission.name}")
            print(f"      → State ID: {state_id}, Expert Choice: {expert_name}")
        
        # Interactive exploration
        print("\n🎮 Interactive Exploration:")
        print("Let's explore what experts recommend in different situations...")
        
        situations = [
            "📱 Smartphone with low battery, high security threat, critical mission",
            "💻 Laptop with full battery, low threat, normal operation", 
            "🔋 Device with critical battery, medium threat, normal mission"
        ]
        
        for situation in situations:
            print(f"\n❓ Situation: {situation}")
            print("   What algorithm would YOU choose? (Think about battery vs security)")
            input("   (Press Enter to see expert recommendation...)")
            
            # Show expert choice with reasoning
            if "low battery" in situation and "high security" in situation:
                print("   🎯 Expert Choice: KYBER (6.2W) - Lowest power but still secure!")
                print("   💡 Reasoning: Critical battery means power saving is priority")
            elif "full battery" in situation and "low threat" in situation:
                print("   🎯 Expert Choice: ASCON (2.1W) - Ultra-efficient for low threats!")  
                print("   💡 Reasoning: Full battery + low threat = optimize for efficiency")
            else:
                print("   🎯 Expert Choice: KYBER (6.2W) - Good balance of security and power")
                print("   💡 Reasoning: Medium situations need balanced approach")
        
        print("\n✅ Key Learning: Each state has different optimal actions!")
        print("   • Battery level affects power budget")
        print("   • Threat level affects security needs") 
        print("   • Mission criticality affects risk tolerance")
    
    def exercise_2_rewards(self):
        """Exercise 2: Understanding the reward function"""
        print("🎯 Goal: Learn how we score different decisions")
        print("\n📐 Our reward function has 3 components:")
        print("   • 40% Battery Efficiency (save power)")
        print("   • 40% Security Match (right protection level)")
        print("   • 20% Expert Agreement (learn from humans)")
        
        # Show reward calculation examples
        print("\n🧮 Reward Calculation Examples:")
        
        test_cases = [
            {
                'scenario': "Critical battery, High threat, Critical mission",
                'state': (BatteryLevel.CRITICAL, ThreatLevel.HIGH, MissionCriticality.CRITICAL),
                'actions_to_test': [0, 4, 7],  # ASCON, KYBER, FALCON
                'action_names': ['ASCON', 'KYBER', 'FALCON']
            }
        ]
        
        for case in test_cases:
            print(f"\n📋 Scenario: {case['scenario']}")
            battery, threat, mission = case['state']
            state_id = self.state_space.encode_state(battery, threat, mission)
            
            # Calculate rewards for different actions
            for i, action in enumerate(case['actions_to_test']):
                print(f"\n   Testing: {case['action_names'][i]}")
                
                # Get reward components (simplified calculation for teaching)
                power_consumption = [2.1, 2.3, 2.5, 2.7, 6.2, 6.5, 6.8, 7.1][action]
                battery_score = max(0, 10 - power_consumption)  # Lower power = higher score
                
                # Security appropriateness (simplified)
                if threat == ThreatLevel.HIGH and action >= 4:  # Post-quantum for high threats
                    security_score = 10
                elif threat == ThreatLevel.LOW and action < 4:  # Pre-quantum for low threats  
                    security_score = 10
                else:
                    security_score = 5
                
                # Expert agreement
                expert_action = self.state_space.get_expert_action(state_id)
                expert_score = 10 if action == expert_action else 0
                
                # Total reward
                total_reward = 0.4 * battery_score + 0.4 * security_score + 0.2 * expert_score
                
                print(f"      🔋 Battery Score: {battery_score:.1f} (power: {power_consumption}W)")  
                print(f"      🛡️  Security Score: {security_score:.1f}")
                print(f"      🎓 Expert Score: {expert_score:.1f}")
                print(f"      📊 Total Reward: {total_reward:.1f}")
        
        print("\n✅ Key Learning: Good actions get high rewards!")
        print("   • Lower power consumption → Higher battery score")
        print("   • Right security level → Higher security score") 
        print("   • Matching expert → Higher expert score")
    
    def exercise_3_decisions(self):
        """Exercise 3: Simulating decision making"""
        print("🎯 Goal: Experience how an RL agent makes decisions")
        print("\n🤖 Let's simulate an RL agent learning to make decisions...")
        
        # Create a mini Q-table for demonstration
        print("📊 Starting with empty knowledge (Q-table):")
        mini_q_table = np.zeros((5, 8))  # 5 states, 8 actions for demo
        print("   All Q-values start at 0.0 (no knowledge)")
        
        # Simulate a few learning episodes
        print("\n🎮 Learning Simulation:")
        demo_experiences = [
            {"state": 0, "action": 4, "reward": 8.5, "description": "Low battery → Choose KYBER (efficient)"},
            {"state": 1, "action": 7, "reward": 6.2, "description": "High threat → Choose FALCON (secure)"},  
            {"state": 0, "action": 7, "reward": 3.1, "description": "Low battery → Choose FALCON (wasteful)"},
            {"state": 2, "action": 0, "description": "Medium threat → Choose ASCON (too weak)", "reward": 4.0}
        ]
        
        learning_rate = 0.1
        
        for step, exp in enumerate(demo_experiences, 1):
            print(f"\n   Step {step}: {exp['description']}")
            
            old_q = mini_q_table[exp['state'], exp['action']]
            new_q = old_q + learning_rate * (exp['reward'] - old_q)
            mini_q_table[exp['state'], exp['action']] = new_q
            
            print(f"      Old Q-value: {old_q:.2f}")
            print(f"      Reward received: {exp['reward']:.1f}")  
            print(f"      New Q-value: {new_q:.2f}")
            print(f"      Learning: {'Good choice!' if exp['reward'] > 6 else 'Poor choice - learned to avoid'}")
        
        print("\n📈 After Learning:")
        print("   The agent now 'knows' which actions work well in each state!")
        print("   High Q-values = Good actions, Low Q-values = Bad actions")
        
        # Show learned preferences
        print("\n🧠 Learned Preferences:")
        action_names = ['ASCON', 'SPECK', 'HIGHT', 'CAMELLIA', 'KYBER', 'DILITHIUM', 'SPHINCS', 'FALCON']
        
        for state in range(3):  # Show first few states
            best_action = np.argmax(mini_q_table[state])
            best_value = mini_q_table[state, best_action]
            if best_value > 0:
                print(f"   State {state}: Prefers {action_names[best_action]} (Q-value: {best_value:.2f})")
        
        print("\n✅ Key Learning: RL agents learn from experience!")
        print("   • Try actions and see what happens")
        print("   • Remember which actions gave good rewards")
        print("   • Gradually become smarter through trial and error")
    
    def exercise_4_learning(self):
        """Exercise 4: Visualizing the learning process"""  
        print("🎯 Goal: See how performance improves over time")
        print("\n📈 Let's create a learning curve visualization...")
        
        # Simulate learning progress
        episodes = 100
        rewards = []
        
        # Simulate realistic learning curve
        np.random.seed(42)  # Reproducible results
        
        print("🏃‍♂️ Simulating 100 episodes of learning...")
        
        for episode in range(episodes):
            if episode < 20:  # Initial exploration phase
                base_reward = 20 + episode * 1.5
                noise = np.random.normal(0, 10)
            elif episode < 50:  # Learning phase  
                base_reward = 50 + (episode - 20) * 0.8
                noise = np.random.normal(0, 8)
            else:  # Convergence phase
                base_reward = 72 + np.random.normal(0, 3)
                noise = np.random.normal(0, 5)
            
            reward = max(0, base_reward + noise)
            rewards.append(reward)
            
            if episode % 20 == 0:
                avg_recent = np.mean(rewards[-10:]) if len(rewards) >= 10 else rewards[-1]
                print(f"   Episode {episode:3d}: Recent avg reward = {avg_recent:.1f}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, alpha=0.6, label='Episode Rewards')
        
        # Add moving average for clearer trend
        window = 10
        moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
        plt.plot(moving_avg, linewidth=2, color='red', label='Moving Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward') 
        plt.title('🎯 RL Agent Learning Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for learning phases
        plt.axvspan(0, 20, alpha=0.2, color='orange', label='Exploration')
        plt.axvspan(20, 50, alpha=0.2, color='yellow', label='Learning')
        plt.axvspan(50, 100, alpha=0.2, color='green', label='Convergence')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = Path("../teaching/visuals/learning_curve_demo.png")
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Learning curve saved to: {plot_path}")
        
        plt.show()
        
        # Analyze the results
        print("\n🔍 Learning Analysis:")
        print(f"   Starting performance: {np.mean(rewards[:10]):.1f}")
        print(f"   Final performance: {np.mean(rewards[-10:]):.1f}")
        print(f"   Improvement: {np.mean(rewards[-10:]) - np.mean(rewards[:10]):.1f} points")
        
        print("\n✅ Key Learning: RL improves through practice!")
        print("   • Phase 1: Random exploration (low, variable rewards)")  
        print("   • Phase 2: Active learning (steadily improving)")
        print("   • Phase 3: Convergence (stable, high performance)")
    
    def exercise_5_comparison(self):
        """Exercise 5: Comparing different approaches"""
        print("🎯 Goal: Understand differences between Q-Learning and Deep Q-Learning")
        print("\n⚖️  Algorithm Comparison:")
        
        comparison_data = {
            'Q-Learning (Tabular)': {
                'description': 'Stores exact Q-values for each state-action pair',
                'pros': ['Simple to understand', 'Exact values', 'Fast for small problems'],
                'cons': ['Memory grows with states', 'Hard to generalize', 'Limited scalability'],
                'best_for': 'Small, well-defined problems (like ours!)',
                'our_performance': '53.9 ± 46.7 average reward'
            },
            'Deep Q-Learning (Neural)': {
                'description': 'Uses neural network to approximate Q-values',  
                'pros': ['Handles large state spaces', 'Can generalize', 'GPU accelerated'],
                'cons': ['More complex', 'Approximate values', 'Requires more data'],
                'best_for': 'Large, complex problems with patterns',
                'our_performance': '52.8 ± 42.9 average reward'
            }
        }
        
        for algo_name, details in comparison_data.items():
            print(f"\n🧠 {algo_name}:")
            print(f"   💡 What it is: {details['description']}")
            print(f"   ✅ Pros: {', '.join(details['pros'])}")
            print(f"   ❌ Cons: {', '.join(details['cons'])}")
            print(f"   🎯 Best for: {details['best_for']}")
            print(f"   📊 Our results: {details['our_performance']}")
        
        print("\n🤔 For Our Problem:")
        print("   • Both algorithms perform similarly (~53 average reward)")
        print("   • Q-Learning is simpler and easier to understand")
        print("   • Deep Q-Learning is more flexible for future extensions")
        print("   • Both achieve expert-level performance quickly")
        
        print("\n🔍 Action Preferences:")
        preferences = {
            'Q-Learning': {'KYBER': '43.3%', 'FALCON': '23.5%', 'DILITHIUM': '15.2%'},
            'Deep Q-Learning': {'KYBER': '48.6%', 'FALCON': '17.0%', 'DILITHIUM': '18.1%'}
        }
        
        for algo, prefs in preferences.items():
            print(f"   {algo}: {', '.join([f'{k} {v}' for k, v in prefs.items()])}")
        
        print("\n✅ Key Learning: Choose the right tool for the job!")
        print("   • Start simple (Q-Learning) for learning and prototyping")
        print("   • Use complex (Deep Q-Learning) for scalability and performance")
        print("   • Both can be excellent choices depending on requirements")

def main():
    """Main interactive learning session"""
    lab = InteractiveLearningLab()
    
    print("\n🎓 Choose your learning path:")
    print("1. 🚀 Run all exercises (full experience)")
    print("2. 🎯 Pick specific exercise")
    print("3. 📚 Quick concept review")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        lab.run_all_exercises()
    elif choice == "2":
        exercises = [
            ("State Space Explorer", lab.exercise_1_state_space),
            ("Reward Calculator", lab.exercise_2_rewards),
            ("Decision Simulator", lab.exercise_3_decisions), 
            ("Learning Visualizer", lab.exercise_4_learning),
            ("Algorithm Comparer", lab.exercise_5_comparison)
        ]
        
        print("\nAvailable exercises:")
        for i, (name, _) in enumerate(exercises, 1):
            print(f"{i}. {name}")
        
        ex_choice = int(input("Choose exercise (1-5): ")) - 1
        if 0 <= ex_choice < len(exercises):
            print(f"\n🎯 Starting: {exercises[ex_choice][0]}")
            exercises[ex_choice][1]()
    
    elif choice == "3":
        print("\n📚 RL Quick Review:")
        print("🤖 Agent: Makes decisions (our RL system)")  
        print("🌍 Environment: The situation (battery + threat + mission)")
        print("📍 State: Current condition (1 of our 30 states)")
        print("⚡ Action: Algorithm choice (1 of 8 crypto algorithms)")
        print("🏆 Reward: Performance score (how good was that choice)")
        print("🧠 Learning: Improve decisions based on experience")
    
    print("\n🎉 Thanks for learning with us!")
    print("💡 You're now ready to explore our RL system!")

if __name__ == "__main__":
    main()
