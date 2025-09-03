# 🧪 Lab 1: Explore the State Space

## 🎯 Lab Objectives
After completing this lab, you will:
- Understand our 30-state system design
- Explore state-action mappings  
- Run expert knowledge queries
- Visualize state space structure
- Understand how states influence algorithm selection

**Estimated Time**: 30-45 minutes  
**Difficulty**: Beginner ⭐⭐⭐

---

## 📋 Prerequisites

Before starting this lab:
- [ ] Read `01_foundations/what_is_reinforcement_learning.md`
- [ ] Read `01_foundations/crypto_algorithms_overview.md`  
- [ ] Have Python environment set up
- [ ] Basic familiarity with running Python scripts

---

## 🚀 Lab Setup

### Step 1: Navigate to Lab Directory
```bash
cd "c:\Users\burak\Desktop\rl-final-crypto\teaching_materials\06_hands_on_labs"
```

### Step 2: Verify Environment
```python
# Run this in Python to verify setup
import sys
import os
sys.path.append('../../src')

try:
    from environment.state_space import StateSpace, CryptoState, CryptoAlgorithm
    print("✅ Lab environment ready!")
except ImportError as e:
    print(f"❌ Setup issue: {e}")
```

---

## 🔍 Part 1: Understanding States (15 minutes)

### Exercise 1.1: Explore State Structure

**Goal**: Understand how our 30 states are organized.

```python
# Create a state space instance
from environment.state_space import StateSpace, CryptoState, CryptoAlgorithm

state_space = StateSpace()

print("🏗️ BATTERY-OPTIMIZED CRYPTO RL - STATE SPACE EXPLORATION")
print("=" * 60)
print(f"Total States: {state_space.total_states}")
print(f"Total Actions: {state_space.total_actions}")
print()

# Let's explore the first 10 states
print("🔍 First 10 States:")
for i in range(10):
    state = CryptoState.from_index(i)
    print(f"State {i:2d}: Battery={state.battery_level}, "
          f"Threat={state.threat_level}, Mission={state.mission_type}")
```

**Expected Output**:
```
🏗️ BATTERY-OPTIMIZED CRYPTO RL - STATE SPACE EXPLORATION
============================================================
Total States: 30
Total Actions: 8

🔍 First 10 States:
State  0: Battery=0, Threat=0, Mission=0
State  1: Battery=0, Threat=0, Mission=1
State  2: Battery=0, Threat=1, Mission=0
...
```

**❓ Questions to Think About**:
1. How many states have Battery=0? (Very Low battery)
2. What's the pattern in state indexing?
3. How many unique combinations of (battery, threat, mission) exist?

### Exercise 1.2: State Interpretation

**Goal**: Understand what states mean in real terms.

```python
# Let's interpret states in human-readable form
def interpret_state(state_index):
    state = CryptoState.from_index(state_index)
    
    battery_names = ["Very Low (0-20%)", "Low (20-40%)", "Medium (40-60%)", 
                    "High (60-80%)", "Very High (80-100%)"]
    threat_names = ["Low Threat", "Medium Threat", "High Threat"]  
    mission_names = ["Normal Mission", "Critical Mission"]
    
    print(f"State {state_index:2d} Interpretation:")
    print(f"  🔋 Battery: {battery_names[state.battery_level]}")
    print(f"  ⚠️  Threat:  {threat_names[state.threat_level]}")  
    print(f"  🎯 Mission: {mission_names[state.mission_type]}")
    print()

# Explore some interesting states
interesting_states = [0, 5, 14, 23, 29]  # Different combinations
print("🎯 INTERESTING STATES TO EXPLORE:")
print("-" * 40)
for state_idx in interesting_states:
    interpret_state(state_idx)
```

**❓ Your Task**: 
1. Run the code above
2. For each state, predict which algorithm might be best
3. Write your predictions in the comments

---

## 🧠 Part 2: Expert Knowledge Exploration (15 minutes)

### Exercise 2.1: Query Expert Decisions

**Goal**: Understand what the expert knowledge recommends for different states.

```python
print("🎓 EXPERT KNOWLEDGE EXPLORATION")
print("=" * 50)

# Function to show expert decision with reasoning
def show_expert_decision(state_index):
    state = CryptoState.from_index(state_index)
    expert_action = state_space.get_expert_action(state)
    
    print(f"State {state_index:2d}: Battery={state.battery_level}, "
          f"Threat={state.threat_level}, Mission={state.mission_type}")
    print(f"  👨‍💼 Expert Choice: {expert_action.name}")
    print(f"  🔍 Algorithm Index: {expert_action.value}")
    
    # Let's add some reasoning based on state characteristics
    if state.battery_level <= 1 and expert_action.value <= 3:
        print("  💡 Reasoning: Low battery → chose efficient pre-quantum algorithm")
    elif state.threat_level >= 2 and expert_action.value >= 4:
        print("  💡 Reasoning: High threat → chose post-quantum algorithm") 
    elif state.mission_type == 1 and expert_action.value >= 4:
        print("  💡 Reasoning: Critical mission → chose secure post-quantum algorithm")
    else:
        print("  💡 Reasoning: Balanced choice based on state conditions")
    print()

# Test expert decisions for various scenarios
test_scenarios = [
    0,   # Very low battery, low threat, normal mission
    5,   # Very low battery, high threat, normal mission  
    14,  # Medium battery, medium threat, normal mission
    23,  # High battery, high threat, normal mission
    29   # Very high battery, high threat, critical mission
]

for scenario in test_scenarios:
    show_expert_decision(scenario)
```

### Exercise 2.2: Expert Decision Patterns

**Goal**: Discover patterns in expert decision making.

```python
print("📊 EXPERT DECISION ANALYSIS")
print("=" * 40)

# Count how often each algorithm is chosen by expert
algorithm_counts = {}
for alg in CryptoAlgorithm:
    algorithm_counts[alg.name] = 0

# Count expert choices across all states  
for state_idx in range(30):
    state = CryptoState.from_index(state_idx)
    expert_action = state_space.get_expert_action(state)
    algorithm_counts[expert_action.name] += 1

print("Expert Algorithm Preferences (out of 30 states):")
for alg_name, count in algorithm_counts.items():
    percentage = (count / 30) * 100
    print(f"  {alg_name:10}: {count:2d} times ({percentage:5.1f}%)")
```

**❓ Analysis Questions**:
1. Which algorithm does the expert prefer most?
2. Which algorithms are rarely chosen? Why might this be?
3. Do you see any patterns related to battery vs security?

---

## 📊 Part 3: State Space Visualization (15 minutes)

### Exercise 3.1: Create State Distribution Charts

**Goal**: Visualize how states are distributed across different conditions.

```python
import matplotlib.pyplot as plt
import numpy as np

# Count states by battery level
battery_counts = [0] * 5  # 5 battery levels
threat_counts = [0] * 3   # 3 threat levels  
mission_counts = [0] * 2  # 2 mission types

for state_idx in range(30):
    state = CryptoState.from_index(state_idx)
    battery_counts[state.battery_level] += 1
    threat_counts[state.threat_level] += 1
    mission_counts[state.mission_type] += 1

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Battery level distribution
battery_labels = ['Very Low', 'Low', 'Med', 'High', 'Very High']
ax1.bar(battery_labels, battery_counts, color='lightblue', alpha=0.7)
ax1.set_title('States by Battery Level')
ax1.set_ylabel('Number of States')

# Threat level distribution
threat_labels = ['Low', 'Medium', 'High']
ax2.bar(threat_labels, threat_counts, color='lightcoral', alpha=0.7)  
ax2.set_title('States by Threat Level')
ax2.set_ylabel('Number of States')

# Mission type distribution
mission_labels = ['Normal', 'Critical']
ax3.bar(mission_labels, mission_counts, color='lightgreen', alpha=0.7)
ax3.set_title('States by Mission Type')  
ax3.set_ylabel('Number of States')

plt.tight_layout()
plt.savefig('state_distribution_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("📊 State distribution chart saved as 'state_distribution_analysis.png'")
```

### Exercise 3.2: Expert Decision Heatmap

**Goal**: Create a heatmap showing expert decisions across all state combinations.

```python
import seaborn as sns

# Create decision matrix: Battery levels x (Threat, Mission) combinations
decision_matrix = np.zeros((5, 6))  # 5 battery x 6 threat-mission combinations

battery_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
combination_labels = []

# Fill the matrix
for battery in range(5):
    col = 0
    for threat in range(3):
        for mission in range(2):
            state = CryptoState(battery, threat, mission)
            expert_action = state_space.get_expert_action(state)
            decision_matrix[battery, col] = expert_action.value
            
            if battery == 0:  # Only create labels once
                threat_name = ['Low', 'Med', 'High'][threat]
                mission_name = ['Norm', 'Crit'][mission]
                combination_labels.append(f'{threat_name}-{mission_name}')
            col += 1

# Create heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(decision_matrix, 
           xticklabels=combination_labels,
           yticklabels=battery_labels,
           annot=True, fmt='.0f', cmap='viridis',
           cbar_kws={'label': 'Algorithm Choice (0-7)'})

plt.title('Expert Decision Heatmap\n(Values are algorithm indices: 0=ASCON, 4=KYBER, etc.)')
plt.xlabel('Threat Level - Mission Type')
plt.ylabel('Battery Level')
plt.tight_layout()
plt.savefig('expert_decision_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

print("🔥 Expert decision heatmap saved as 'expert_decision_heatmap.png'")
```

**❓ Heatmap Analysis**:
1. What patterns do you see in the heatmap?
2. How does battery level influence algorithm choice?  
3. Which combinations lead to post-quantum algorithm selection?

---

## 🧪 Part 4: Interactive Exploration (Optional - 10 minutes)

### Exercise 4.1: Custom State Queries

**Goal**: Explore specific states you're curious about.

```python
def interactive_state_explorer():
    print("\n🔍 INTERACTIVE STATE EXPLORER")
    print("Enter battery (0-4), threat (0-2), mission (0-1) to explore states")
    print("Or type 'quit' to exit")
    
    while True:
        try:
            user_input = input("\nEnter state (battery,threat,mission) or 'quit': ").strip()
            if user_input.lower() == 'quit':
                break
                
            # Parse input
            parts = user_input.split(',')
            if len(parts) != 3:
                print("Please enter in format: battery,threat,mission (e.g., 2,1,0)")
                continue
                
            battery, threat, mission = map(int, parts)
            
            # Validate ranges
            if not (0 <= battery <= 4 and 0 <= threat <= 2 and 0 <= mission <= 1):
                print("Valid ranges: battery(0-4), threat(0-2), mission(0-1)")
                continue
            
            # Create and analyze state
            state = CryptoState(battery, threat, mission)
            state_idx = state.to_index()
            expert_action = state_space.get_expert_action(state)
            
            print(f"\n📊 State Analysis:")
            print(f"   State Index: {state_idx}")
            print(f"   Battery: {['Very Low', 'Low', 'Medium', 'High', 'Very High'][battery]}")
            print(f"   Threat: {['Low', 'Medium', 'High'][threat]}")
            print(f"   Mission: {['Normal', 'Critical'][mission]}")
            print(f"   Expert Choice: {expert_action.name} (index {expert_action.value})")
            
        except ValueError:
            print("Please enter valid integers")
        except Exception as e:
            print(f"Error: {e}")

# Run interactive explorer
interactive_state_explorer()
```

### Exercise 4.2: Algorithm Preference Analysis

**Goal**: Understand when each algorithm is preferred.

```python
print("\n🎯 ALGORITHM PREFERENCE ANALYSIS")
print("-" * 50)

# For each algorithm, find states where it's the expert choice
for algorithm in CryptoAlgorithm:
    states_using_alg = []
    
    for state_idx in range(30):
        state = CryptoState.from_index(state_idx)
        expert_action = state_space.get_expert_action(state)
        if expert_action == algorithm:
            states_using_alg.append(state_idx)
    
    if states_using_alg:
        print(f"\n{algorithm.name} (Algorithm {algorithm.value}):")
        print(f"  Used in {len(states_using_alg)} states: {states_using_alg}")
        
        # Analyze conditions where this algorithm is chosen
        battery_levels = []
        threat_levels = []
        mission_types = []
        
        for state_idx in states_using_alg:
            state = CryptoState.from_index(state_idx)
            battery_levels.append(state.battery_level)
            threat_levels.append(state.threat_level)
            mission_types.append(state.mission_type)
        
        avg_battery = np.mean(battery_levels)
        avg_threat = np.mean(threat_levels)
        critical_missions = sum(mission_types)
        
        print(f"  Avg Battery Level: {avg_battery:.1f} ({'Low' if avg_battery < 2 else 'High'} battery scenarios)")
        print(f"  Avg Threat Level: {avg_threat:.1f} ({'Low' if avg_threat < 1 else 'High'} threat scenarios)")
        print(f"  Critical Missions: {critical_missions}/{len(states_using_alg)} ({100*critical_missions/len(states_using_alg):.0f}%)")
```

---

## 📝 Lab Reflection & Questions

### Reflection Questions:
1. **State Space Understanding**:
   - How well does our 30-state design capture the problem complexity?
   - Are there any important scenarios missing?

2. **Expert Knowledge**:
   - Do the expert decisions make intuitive sense?
   - Where might the expert knowledge be conservative or aggressive?

3. **Algorithm Selection Patterns**:
   - Which algorithms seem to be "specialists" for specific conditions?
   - Which are "generalists" used across many states?

4. **Battery vs Security Trade-offs**:
   - How does the expert balance efficiency vs security?
   - Are there surprising choices that challenge your assumptions?

### Discussion Points for Team:
1. Would you change any expert decisions? Which ones and why?
2. How might real-world conditions differ from our state representation?
3. What additional factors might influence algorithm selection?

---

## ✅ Lab Completion Checklist

- [ ] **Part 1**: Understood 30-state structure and indexing
- [ ] **Part 2**: Explored expert knowledge and decision patterns  
- [ ] **Part 3**: Created visualizations of state space and decisions
- [ ] **Part 4**: Completed interactive exploration (optional)
- [ ] **Reflection**: Answered key questions about state-action mappings

### Generated Files:
- [ ] `state_distribution_analysis.png` - Bar charts showing state distributions
- [ ] `expert_decision_heatmap.png` - Heatmap of expert choices

---

## 🚀 Next Steps

**Congratulations!** You've successfully explored our state space design.

**Next Lab**: `lab2_run_q_learning.md` - Train your first Q-Learning agent

**Key Insights You Should Have**:
1. Our 30 states systematically cover important battery-threat-mission combinations
2. Expert knowledge shows clear patterns: low battery → efficient algorithms, high threat → post-quantum algorithms  
3. Algorithm selection involves complex trade-offs that RL can learn to optimize

**Questions for Next Lab**:
- How will Q-Learning improve upon expert decisions?
- Which states will be hardest for RL to learn optimal policies for?
- How will training dynamics evolve over episodes?

---

## 🆘 Troubleshooting

### Common Issues:

**Problem**: `ImportError` when importing modules  
**Solution**: Check your Python path and make sure you're running from correct directory

**Problem**: Plots don't display  
**Solution**: Make sure matplotlib backend is set up correctly: `plt.show()` vs `plt.savefig()`

**Problem**: Expert decisions seem wrong  
**Solution**: Remember indices: ASCON=0, SPECK=1, ..., FALCON=7. Check algorithm mapping.

### Getting Help:
- Review `../quick_reference/troubleshooting.md`
- Ask teammates about confusing expert decisions
- Check that all prerequisite readings are completed

**Happy Exploring!** 🧪
