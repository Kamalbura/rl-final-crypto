# Reinforcement Learning for Cryptographic Algorithm Selection
## Complete System Architecture & Documentation

**Project**: Battery-Optimized Cryptographic Selection using Reinforcement Learning  
**Date**: September 4, 2025  
**Organization**: RL Research Team  
**Phase**: 1 - System Architecture Documentation

---

## 🏗️ System Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RL CRYPTO SELECTION SYSTEM              │
├─────────────────────────────────────────────────────────────┤
│  🤖 RL AGENTS                                              │
│  ├─ Q-Learning Agent (Tabular)                             │
│  ├─ Deep Q-Network (DQN) Agent                             │
│  └─ Expert Knowledge Integration                           │
├─────────────────────────────────────────────────────────────┤
│  🌍 ENVIRONMENT                                            │
│  ├─ CryptoEnvironment                                      │
│  ├─ StateSpace Management                                  │
│  └─ Reward Function Engine                                 │
├─────────────────────────────────────────────────────────────┤
│  📊 DATA & MONITORING                                      │
│  ├─ Training Analytics                                     │
│  ├─ Performance Metrics                                    │
│  └─ Visualization System                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 State Space Definition

### State Components (Total: 30 States)
Our system models real-world device conditions through three dimensions:

#### 1. Battery Level (5 Levels)
- **CRITICAL** (0-20%): Extreme power conservation needed
- **LOW** (20-40%): Significant power constraints  
- **MEDIUM** (40-60%): Moderate power awareness
- **HIGH** (60-80%): Good power availability
- **FULL** (80-100%): Maximum power available

#### 2. Threat Level (3 Levels)
- **LOW**: Basic security sufficient (internal networks)
- **MEDIUM**: Moderate security required (public networks)
- **HIGH**: Maximum security critical (sensitive operations)

#### 3. Mission Criticality (2 Levels)
- **NORMAL**: Standard operations, efficiency preferred
- **CRITICAL**: Mission-critical, reliability paramount

### State Encoding Formula
```
State_ID = Battery_Level * 6 + Threat_Level * 2 + Mission_Type
Range: 0-29 (30 unique states)
```

### Example States
| State ID | Battery | Threat | Mission | Expert Choice | Reasoning |
|----------|---------|--------|---------|---------------|-----------|
| 0 | CRITICAL | LOW | NORMAL | SPECK | Ultra-low power needed |
| 14 | MEDIUM | HIGH | NORMAL | KYBER | Balance security/power |
| 29 | FULL | HIGH | CRITICAL | FALCON | Maximum security available |

---

## 🎮 Action Space Definition

### Available Actions (8 Cryptographic Algorithms)

#### Pre-Quantum Algorithms (Actions 0-3)
| Action | Algorithm | Power (W) | Security Level | Use Case |
|--------|-----------|-----------|----------------|----------|
| 0 | ASCON | 2.1 | Lightweight | IoT, sensors |
| 1 | SPECK | 2.3 | Fast | Real-time systems |
| 2 | HIGHT | 2.5 | Moderate | Mobile devices |
| 3 | CAMELLIA | 2.7 | Traditional | Legacy systems |

#### Post-Quantum Algorithms (Actions 4-7)
| Action | Algorithm | Power (W) | Security Level | Use Case |
|--------|-----------|-----------|----------------|----------|
| 4 | KYBER | 6.2 | PQ-Secure | General purpose |
| 5 | DILITHIUM | 6.5 | PQ-Signatures | Digital signing |
| 6 | SPHINCS | 6.8 | High Security | Critical systems |
| 7 | FALCON | 7.1 | Balanced | Enterprise |

### Algorithm Selection Strategy
- **Low Threat**: Pre-quantum algorithms preferred for efficiency
- **High Threat**: Post-quantum algorithms required for security
- **Critical Battery**: Lowest power consumption prioritized
- **Full Battery**: Security level takes precedence

---

## 🏆 Reward Function Architecture

### Multi-Component Reward System
```
Total_Reward = 0.4 * Battery_Efficiency + 
               0.4 * Security_Appropriateness + 
               0.2 * Expert_Agreement
```

#### Component 1: Battery Efficiency (40% Weight)
```python
def calculate_battery_reward(state, action):
    power_consumption = algorithm_power[action]
    battery_capacity = battery_levels[state.battery_level]
    
    # Reward efficiency, penalize waste
    efficiency = min(10, battery_capacity / power_consumption)
    return efficiency * 2 - 10  # Range: [-10, +10]
```

**Examples**:
- SPECK (2.3W) on FULL battery: +10 (excellent efficiency)
- FALCON (7.1W) on CRITICAL battery: -8 (power waste)

#### Component 2: Security Appropriateness (40% Weight)
```python
def calculate_security_reward(state, action):
    threat = state.threat_level
    is_post_quantum = action >= 4
    
    if threat == HIGH and is_post_quantum:
        return +10  # Perfect match
    elif threat == LOW and not is_post_quantum:
        return +8   # Good match
    elif threat == MEDIUM:
        return +6   # Acceptable
    else:
        return -5   # Mismatch
```

#### Component 3: Expert Agreement (20% Weight)
```python
def calculate_expert_reward(action, expert_action):
    if action == expert_action:
        return +10  # Perfect agreement
    else:
        # Partial credit for similar power consumption
        power_diff = abs(power[action] - power[expert_action])
        if power_diff <= 0.3: return +5
        elif power_diff <= 0.6: return +2
        else: return -2
```

### Reward Distribution Examples

| Scenario | Algorithm | Battery | Security | Expert | Total | Grade |
|----------|-----------|---------|----------|---------|-------|-------|
| Critical+High+Critical | KYBER | 6.0 | 10.0 | 10.0 | 10.4 | A |
| Full+Low+Normal | ASCON | 8.0 | 8.0 | 10.0 | 10.4 | A |
| Critical+High+Normal | FALCON | -2.0 | 10.0 | 5.0 | 5.2 | C |
| Full+Low+Normal | FALCON | 2.0 | 2.0 | 0.0 | 1.6 | D |

---

## 🧠 Learning Algorithms

### 1. Q-Learning Agent (Tabular Method)

#### Architecture
```python
Q_Table: shape = (30 states, 8 actions)
Memory Requirements: 240 values
Update Rule: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

#### Hyperparameters
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.95  
- **Epsilon (exploration)**: 0.1 → 0.01 (decay)
- **Episodes**: 10,000

#### Advantages
- ✅ Guaranteed convergence
- ✅ Interpretable results  
- ✅ No overfitting risk
- ✅ Fast inference

#### Limitations
- ❌ Limited to small state spaces
- ❌ No generalization
- ❌ Manual feature engineering

### 2. Deep Q-Network (DQN) Agent

#### Neural Network Architecture
```
Input Layer: 7 features (state encoding)
├─ Battery Level (one-hot: 5 features)
├─ Threat Level (scalar: 1 feature)  
└─ Mission Type (scalar: 1 feature)

Hidden Layer 1: 64 neurons (ReLU)
Hidden Layer 2: 32 neurons (ReLU)
Hidden Layer 3: 16 neurons (ReLU)

Output Layer: 8 neurons (Q-values for each action)
```

#### Training Features
- **Experience Replay**: 10,000 transitions
- **Target Network**: Updated every 100 steps
- **Double DQN**: Reduced overestimation bias
- **Epsilon-Greedy**: 1.0 → 0.01 over 1000 episodes

#### Advantages
- ✅ Scales to larger problems
- ✅ Automatic feature learning
- ✅ Better generalization
- ✅ Handles continuous states

#### Training Process
1. **Data Collection**: Agent explores environment
2. **Experience Storage**: (state, action, reward, next_state)
3. **Batch Training**: Random samples from replay buffer
4. **Network Updates**: Minimize TD-error loss

---

## 📊 Expert Knowledge Integration

### Lookup Table System
Our system incorporates human expert knowledge through a comprehensive lookup table:

```python
# Example expert mappings
expert_knowledge = {
    (CRITICAL, HIGH, CRITICAL): "KYBER",     # Balance power/security
    (FULL, LOW, NORMAL): "ASCON",            # Maximize efficiency  
    (LOW, HIGH, CRITICAL): "KYBER",          # Security critical
    (MEDIUM, MEDIUM, NORMAL): "SPECK",       # Balanced approach
    # ... 26 more mappings
}
```

### Knowledge Validation
- Expert recommendations tested in simulation
- Performance verified against optimal policies
- Continuous refinement based on results

---

## 🔄 Training Pipeline

### Phase 1: Environment Setup
1. **State Space Initialization**: 30 states configured
2. **Action Space Setup**: 8 algorithms loaded
3. **Reward Function Calibration**: Components weighted and tested
4. **Expert Knowledge Loading**: Lookup table validated

### Phase 2: Agent Training
```python
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    # Record metrics
    episode_rewards.append(total_reward)
    if episode % 100 == 0:
        evaluate_performance()
```

### Phase 3: Performance Evaluation
- **Training Metrics**: Reward progression, convergence
- **Testing Metrics**: Success rate, efficiency, security compliance
- **Comparative Analysis**: Q-Learning vs DQN performance

---

## 📈 Monitoring & Analytics

### Key Performance Indicators (KPIs)

#### 1. Learning Efficiency
- **Convergence Rate**: Episodes to reach 90% optimal performance
- **Sample Efficiency**: Reward per training sample
- **Stability**: Variance in final performance

#### 2. Decision Quality  
- **Expert Agreement**: % alignment with human experts
- **Security Compliance**: % appropriate security level selections
- **Power Efficiency**: Average power consumption vs optimal

#### 3. Robustness
- **State Coverage**: % of states explored during training
- **Action Diversity**: Distribution of actions taken
- **Generalization**: Performance on unseen state distributions

---

## 🎯 Production Deployment

### System Requirements
- **Memory**: 2GB RAM minimum
- **CPU**: Intel i5 or equivalent
- **Python**: 3.8+ with required dependencies
- **Storage**: 1GB for logs and models

### Integration Points
1. **Input Interface**: Device state monitoring
2. **Decision Engine**: Trained RL agents
3. **Output Interface**: Algorithm selection commands
4. **Monitoring**: Real-time performance tracking

### Deployment Architecture
```
Device Sensors → State Estimation → RL Agent → Algorithm Selection → Crypto System
                        ↓
                 Performance Monitoring → Continuous Learning
```

---

## 📝 Documentation Standards

### Code Documentation
- **Docstrings**: All functions and classes documented
- **Type Hints**: Full type annotation coverage
- **Comments**: Complex logic explained inline
- **Examples**: Usage examples for key components

### Research Documentation  
- **Methodology**: Detailed experimental design
- **Results**: Comprehensive performance analysis
- **Reproducibility**: Complete parameter specifications
- **Validation**: Independent verification procedures

---

*End of Phase 1 Documentation*

**Next Phase**: Training & Validation Results with Visualizations
**Progress**: 20% Complete - Architecture Documentation Finished
