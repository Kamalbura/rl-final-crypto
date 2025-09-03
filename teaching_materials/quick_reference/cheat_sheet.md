# 📚 Quick Reference Guide - Battery-Optimized Crypto RL

## 🚀 Quick Start Commands

### Essential Commands
```bash
# Navigate to project
cd "c:\Users\burak\Desktop\rl-final-crypto"

# Run basic Q-Learning
python src/algorithms/q_learning.py

# Run DQN with CUDA
python src/algorithms/deep_q_learning.py

# Advanced training suite
python src/advanced_training.py

# Generate visualizations
python src/visualization_system.py
```

### Training Pipeline
```bash
# Complete training pipeline
python src/training_pipeline.py

# Extended experiments (1000+ episodes)
python src/advanced_training.py --complete-suite
```

---

## 🏗️ System Architecture

### Core Components
```
rl-final-crypto/
├── src/
│   ├── environment/           # State space & crypto environment
│   ├── algorithms/            # Q-Learning & DQN implementations
│   ├── advanced_training.py   # Extended experiments
│   └── visualization_system.py # Advanced plots
├── data/
│   └── lookup_table_*.xlsx    # Expert knowledge
├── results/                   # Training outputs
└── teaching_materials/        # Educational content
```

### Key Classes
- **StateSpace**: Manages 30-state system
- **CryptoEnvironment**: RL environment with battery & crypto simulation  
- **QLearningAgent**: Traditional tabular Q-Learning
- **DQNAgent**: Deep Q-Network with neural networks

---

## 🎯 State Space Reference

### 30 States = 5 × 3 × 2
```
Battery Levels (5):  0=Very Low, 1=Low, 2=Medium, 3=High, 4=Very High
Threat Levels (3):   0=Low, 1=Medium, 2=High  
Mission Types (2):   0=Normal, 1=Critical

State Index = battery + (threat × 5) + (mission × 15)
```

### Algorithm Actions (8)
```
0: ASCON      (Efficient lightweight)
1: SPECK      (Very fast, low power)
2: HIGHT      (Balanced performance)
3: CAMELLIA   (Traditional secure)
4: KYBER      (Post-quantum efficient)  
5: DILITHIUM  (Post-quantum signatures)
6: SPHINCS    (Ultra-secure)
7: FALCON     (Fast post-quantum)
```

---

## ⚡ Power Consumption Hierarchy

```
Most Efficient → Least Efficient
ASCON < SPECK < HIGHT < KYBER < CAMELLIA < DILITHIUM < FALCON < SPHINCS
```

---

## 🏆 Reward Function

```
Total Reward = Battery Efficiency (40%) + 
               Security Appropriateness (40%) + 
               Expert Agreement Bonus (20%)

Range: -50 to +50
Good performance: > +20
Excellent performance: > +40
```

---

## 🧠 Algorithm Comparison

| Aspect | Q-Learning | DQN |
|--------|------------|-----|
| **Learning** | Q-table lookup | Neural network |
| **Memory** | O(states × actions) | O(network parameters) |
| **Generalization** | None | Good |
| **Training Speed** | Fast | Moderate |
| **Scalability** | Poor | Excellent |
| **Interpretability** | High | Low |

---

## 📊 Key Metrics

### Training Metrics
- **Average Reward**: Target > 30
- **Expert Agreement**: Target > 80%
- **Convergence**: Within 500 episodes
- **Exploration Rate**: Start 0.9 → End 0.01

### Evaluation Metrics
- **Average Episode Reward**
- **Standard Deviation of Rewards**
- **Expert Agreement Percentage**  
- **Algorithm Usage Distribution**

---

## 🔧 Hyperparameter Defaults

### Q-Learning
```python
learning_rate = 0.1
discount_factor = 0.95  
epsilon_start = 0.3
epsilon_end = 0.01
epsilon_decay = 0.9995
```

### DQN
```python
learning_rate = 0.0005
batch_size = 64
buffer_capacity = 50000
target_update_frequency = 100
hidden_sizes = [256, 128, 64]
```

---

## 🎨 Visualization Outputs

### Static Plots
- **Training curves**: Reward over episodes
- **Expert agreement**: Agreement percentage over time
- **State space analysis**: 3D and heatmap visualizations
- **Algorithm comparison**: Performance comparisons

### Interactive Plots  
- **Q-value explorer**: Interactive heatmaps (HTML)
- **Training animations**: Dynamic learning progress (GIF)
- **Dashboard**: Comprehensive comparison (HTML)

---

## 🚨 Common Issues & Solutions

### Import Errors
```python
# Add paths if needed
import sys
sys.path.append('../../src')
```

### CUDA Issues
```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Training Too Slow
- Reduce episodes for testing: `num_episodes=100`
- Use smaller networks: `hidden_sizes=[64, 32]`
- Increase evaluation frequency: `evaluation_frequency=50`

### Poor Performance
- Check warm-start: `warm_start=True`
- Verify reward function
- Check expert knowledge integration

---

## 📈 Expected Results

### Q-Learning (1000 episodes)
- **Final Reward**: 50-70
- **Expert Agreement**: 75-85%
- **Training Time**: 2-5 minutes
- **Convergence**: ~300-500 episodes

### DQN (1000 episodes)  
- **Final Reward**: 40-60
- **Expert Agreement**: 70-80%
- **Training Time**: 5-15 minutes (with GPU)
- **Convergence**: ~400-600 episodes

---

## 🔍 Debugging Checklist

### Training Issues
- [ ] Verify expert knowledge loaded correctly
- [ ] Check state-action space dimensions (30×8)
- [ ] Ensure reward function working
- [ ] Confirm environment step function

### Performance Issues
- [ ] Warm-start enabled?
- [ ] Appropriate hyperparameters?
- [ ] Sufficient training episodes?
- [ ] Exploration-exploitation balance?

### Technical Issues  
- [ ] Python path correct?
- [ ] Dependencies installed?
- [ ] GPU memory sufficient (if using CUDA)?
- [ ] Results directory exists and writable?

---

## 💡 Pro Tips

### For Better Performance
1. **Use warm-start**: Always enable expert knowledge initialization
2. **Tune exploration**: Balance exploration vs exploitation
3. **Monitor agreement**: Track expert agreement as key metric
4. **Visualize progress**: Use plots to understand learning dynamics

### For Development
1. **Test with fewer episodes**: Use 100-200 episodes for rapid testing
2. **Save models**: Always save trained models for reuse
3. **Log everything**: Comprehensive logging helps debugging
4. **Compare algorithms**: Run both Q-Learning and DQN for comparison

### For Teaching
1. **Start simple**: Begin with state space exploration
2. **Use visualizations**: Plots make concepts clearer
3. **Interactive demos**: Hands-on labs are most effective
4. **Real examples**: Connect to practical crypto applications

---

## 📞 Quick Help

### File Locations
- **Main algorithms**: `src/algorithms/`
- **Results**: `results/`
- **Teaching materials**: `teaching_materials/`
- **Expert knowledge**: `data/lookup_table_*.xlsx`

### Key Functions
```python
# Load state space
state_space = StateSpace()

# Create environment
env = CryptoEnvironment()

# Train Q-Learning
agent = QLearningAgent(state_space=state_space, warm_start=True)
agent.train(env, num_episodes=1000)

# Train DQN  
agent = DQNAgent(state_space=state_space, warm_start=True)
agent.train(env, num_episodes=1000)
```

### Quick Tests
```python
# Test single episode
obs = env.reset()
action = agent.act(obs)
next_obs, reward, done, info = env.step(action)

# Check expert knowledge
state = CryptoState.from_index(0)
expert_action = state_space.get_expert_action(state)
```

---

**Need More Help?**
- 📖 Check full documentation in `teaching_materials/`
- 🧪 Try hands-on labs in `06_hands_on_labs/`
- 🎨 Run visualization system for insights
- 🔬 Use advanced training for comprehensive analysis

**Happy Learning & Training!** 🚀
