# 🔐 Cryptographic Algorithms Overview

## 🎯 Learning Objectives
After reading this guide, you will understand:
- The 8 cryptographic algorithms our system chooses between
- Difference between pre-quantum and post-quantum cryptography  
- Power consumption and security trade-offs
- Why algorithm selection matters for battery-powered devices

---

## 🤔 Why Do We Need Different Crypto Algorithms?

Imagine you have a toolbox with different tools. You wouldn't use a sledgehammer to tighten a screw, and you wouldn't use a screwdriver to break down a wall. Similarly, different cryptographic algorithms are optimized for different situations:

- **Battery Life vs Security**: Some algorithms use less power but provide less security
- **Speed vs Strength**: Some are faster but easier to break
- **Current vs Future Threats**: Some resist today's attacks, others resist future quantum computers

Our RL system learns to pick the **right tool for the right job** based on current conditions.

---

## 🏛️ The Two Categories: Pre-Quantum vs Post-Quantum

### 🔒 Pre-Quantum Cryptography (Traditional)
**What it is**: Cryptographic algorithms designed to resist attacks from classical computers.

**Strengths**:
✅ Well-tested and trusted  
✅ Very efficient (low power, fast)  
✅ Widely supported in hardware  
✅ Smaller key and message sizes  

**Weaknesses**:
❌ Vulnerable to future quantum computers  
❌ May become obsolete in 10-15 years  
❌ Not suitable for high-security, long-term applications  

### 🛡️ Post-Quantum Cryptography (Future-Proof)  
**What it is**: Cryptographic algorithms designed to resist attacks from both classical AND quantum computers.

**Strengths**:
✅ Secure against quantum computer attacks  
✅ Future-proof for long-term security  
✅ Being standardized by NIST  
✅ Essential for critical infrastructure  

**Weaknesses**:
❌ Higher power consumption  
❌ Larger key sizes and messages  
❌ Less mature (newer, less tested)  
❌ May require more computational resources  

---

## 🔧 Our 8 Cryptographic Algorithms

### 📱 Pre-Quantum Algorithms (4 algorithms)

#### 1. 🟢 ASCON (Lightweight Champion)
```
Power Consumption: ⚡⚡ (Very Low)
Security Level:    🔒🔒🔒 (Good)  
Speed:            🚀🚀🚀🚀 (Very Fast)
Use Case:         IoT devices, sensors
```

**What it is**: Winner of NIST Lightweight Cryptography competition  
**Best for**: Battery-critical situations with moderate security needs  
**Real-world example**: Smart sensors that need to run for years on one battery  

**Why RL might choose it**:
- When battery is very low
- When threat level is low to moderate
- When mission allows slightly reduced security for much longer battery life

#### 2. 🟡 SPECK (Speed Demon)
```
Power Consumption: ⚡⚡ (Very Low)
Security Level:    🔒🔒 (Moderate)
Speed:            🚀🚀🚀🚀🚀 (Extremely Fast)  
Use Case:         High-throughput, low-power applications
```

**What it is**: NSA-designed algorithm optimized for software performance  
**Best for**: Applications where speed is critical and power is limited  
**Real-world example**: Military communications in resource-constrained environments  

**Why RL might choose it**:
- When processing large amounts of data quickly
- When battery level is low
- When threat environment is relatively controlled

#### 3. 🟠 HIGHT (Balanced Performer)
```
Power Consumption: ⚡⚡⚡ (Low-Medium)
Security Level:    🔒🔒🔒 (Good)
Speed:            🚀🚀🚀 (Fast)
Use Case:         General-purpose applications
```

**What it is**: Korean standard designed for low-power devices  
**Best for**: Balanced applications needing decent security and efficiency  
**Real-world example**: RFID systems, smart cards  

**Why RL might choose it**:
- When need balance between security and efficiency
- When battery level is medium
- When threat level is moderate

#### 4. 🔴 CAMELLIA (Traditional Strength)
```
Power Consumption: ⚡⚡⚡⚡ (Medium-High)
Security Level:    🔒🔒🔒🔒 (High)
Speed:            🚀🚀 (Moderate)
Use Case:         Traditional enterprise applications
```

**What it is**: International standard, equivalent to AES in security  
**Best for**: Applications where security is paramount and power is available  
**Real-world example**: Banking systems, secure communications  

**Why RL might choose it**:
- When battery level is high  
- When security requirements are strict
- When can afford higher power consumption

### 🛡️ Post-Quantum Algorithms (4 algorithms)

#### 5. 🟢 KYBER (Post-Quantum Efficiency Leader)
```
Power Consumption: ⚡⚡⚡ (Low-Medium)
Security Level:    🔒🔒🔒🔒🔒 (Very High, Quantum-Resistant)
Speed:            🚀🚀🚀 (Fast)
Use Case:         Key exchange for post-quantum security
```

**What it is**: NIST standard for post-quantum key encapsulation  
**Best for**: Establishing secure communications in post-quantum world  
**Real-world example**: Secure messaging apps preparing for quantum threats  

**Why RL might choose it**:
- When post-quantum security is required
- When battery level allows moderate consumption
- When establishing secure channels

#### 6. 🟡 DILITHIUM (Digital Signature Specialist)
```
Power Consumption: ⚡⚡⚡⚡ (Medium-High)
Security Level:    🔒🔒🔒🔒🔒 (Very High, Quantum-Resistant)
Speed:            🚀🚀 (Moderate)
Use Case:         Digital signatures and authentication
```

**What it is**: NIST standard for post-quantum digital signatures  
**Best for**: Authenticating messages and documents  
**Real-world example**: Secure software updates, digital contracts  

**Why RL might choose it**:
- When digital signatures are needed
- When battery level is medium to high
- When long-term authenticity is critical

#### 7. 🟠 SPHINCS+ (Ultra-Secure Signatures)
```
Power Consumption: ⚡⚡⚡⚡⚡ (High)
Security Level:    🔒🔒🔒🔒🔒 (Very High, Quantum-Resistant)
Speed:            🚀 (Slow)
Use Case:         Ultra-high security applications
```

**What it is**: Hash-based signatures with strongest security guarantees  
**Best for**: Critical infrastructure and highest-security applications  
**Real-world example**: Nuclear facility communications, national security  

**Why RL might choose it**:
- When security is absolutely paramount
- When battery level is high
- When can tolerate slower performance for maximum security

#### 8. 🔴 FALCON (Fast Lattice Signatures)
```
Power Consumption: ⚡⚡⚡⚡ (Medium-High)
Security Level:    🔒🔒🔒🔒🔒 (Very High, Quantum-Resistant)
Speed:            🚀🚀🚀 (Fast)
Use Case:         High-performance post-quantum signatures
```

**What it is**: Lattice-based signatures with good performance  
**Best for**: High-throughput applications needing post-quantum security  
**Real-world example**: Real-time secure communications  

**Why RL might choose it**:
- When need fast post-quantum signatures
- When battery level is sufficient
- When balancing performance and post-quantum security

---

## ⚡ Power Consumption Hierarchy

Understanding power consumption is crucial for our battery optimization:

### Most Efficient (Best for Low Battery) ➔ Least Efficient
```
1. ASCON     ⚡⚡ (2.1W avg)
2. SPECK     ⚡⚡ (1.8W avg)  
3. HIGHT     ⚡⚡⚡ (2.5W avg)
4. KYBER     ⚡⚡⚡ (3.2W avg)
5. CAMELLIA  ⚡⚡⚡⚡ (4.1W avg)
6. DILITHIUM ⚡⚡⚡⚡ (4.8W avg)
7. FALCON    ⚡⚡⚡⚡⚡ (5.2W avg)  
8. SPHINCS+  ⚡⚡⚡⚡⚡ (5.8W avg)
```

**Key Insight**: There's a general trade-off between power efficiency and advanced security features. Post-quantum algorithms typically consume more power due to:
- Larger key sizes
- More complex mathematical operations  
- Additional computational overhead

---

## 🛡️ Security Comparison

### Against Classical Attacks:
**All 8 algorithms provide strong security against current computers**

### Against Quantum Attacks:
- **Pre-quantum (ASCON, SPECK, HIGHT, CAMELLIA)**: ❌ Vulnerable
- **Post-quantum (KYBER, DILITHIUM, SPHINCS+, FALCON)**: ✅ Resistant  

### Security Timeline:
```
Today ─────────────────── 2030 ─────────────── 2035+ ─────────────── Future
  │                         │                    │
  │ All algorithms secure   │ Quantum threat     │ Post-quantum only
  │ against classical       │ becomes real       │ algorithms secure
  │ computers               │                    │
```

---

## 🎯 Algorithm Selection Strategy

Our RL system learns to balance these factors:

### 🔋 Battery Level Decision Tree:
```
Very Low Battery (0-20%):
├── Low Threat: SPECK or ASCON
└── High Threat: ASCON (best efficiency with decent security)

Medium Battery (20-60%):  
├── Low Threat: HIGHT or KYBER
├── Medium Threat: KYBER or DILITHIUM
└── High Threat: DILITHIUM or FALCON

High Battery (60-100%):
├── Normal Mission: CAMELLIA or FALCON
└── Critical Mission: SPHINCS+ or DILITHIUM
```

### 🎖️ Mission Type Influence:
- **Normal Mission**: Can accept some security trade-offs for efficiency
- **Critical Mission**: Security is paramount, efficiency is secondary

### ⚠️ Threat Level Adaptation:
- **Low Threat**: Pre-quantum algorithms acceptable  
- **Medium Threat**: Prefer post-quantum but consider efficiency
- **High Threat**: Post-quantum algorithms mandatory

---

## 🧪 Real-World Scenarios

Let's see how different scenarios would influence algorithm choice:

### Scenario 1: Emergency Communication Device
```
State: (Very Low Battery, High Threat, Critical Mission)
Best Choice: ASCON
Reasoning: Battery about to die, need any communication possible
RL Learning: Even in high threat, sometimes efficiency trumps security
```

### Scenario 2: Secure Military Base
```
State: (High Battery, High Threat, Critical Mission)  
Best Choice: SPHINCS+
Reasoning: Maximum security needed, power available
RL Learning: When resources allow, prioritize strongest security
```

### Scenario 3: IoT Sensor Network
```
State: (Medium Battery, Low Threat, Normal Mission)
Best Choice: KYBER
Reasoning: Future-proof security with reasonable efficiency
RL Learning: Balance current needs with future requirements
```

### Scenario 4: Mobile Reconnaissance  
```
State: (Low Battery, Medium Threat, Normal Mission)
Best Choice: HIGHT or KYBER
Reasoning: Need reliability without draining battery
RL Learning: Medium threat requires post-quantum consideration
```

---

## 📊 Performance Comparison Matrix

| Algorithm | Power | Speed | Classical Security | Quantum Security | Key Size | Best Use Case |
|-----------|-------|-------|-------------------|------------------|----------|---------------|
| ASCON     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | Small | Battery critical |
| SPECK     | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ❌ | Small | Speed critical |
| HIGHT     | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | Medium | Balanced |
| CAMELLIA  | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ❌ | Medium | Traditional secure |
| KYBER     | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Large | Future key exchange |
| DILITHIUM | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ✅ | Large | Future signatures |
| SPHINCS+  | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ✅ | Large | Maximum security |
| FALCON    | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Large | Fast post-quantum |

---

## 🧠 Why RL is Perfect for This Problem

### 🔄 **Dynamic Optimization**
Traditional systems use fixed rules: "If battery < 20%, use ASCON"

RL learns nuanced strategies: "If battery is 25% but threat is high and mission is critical, KYBER might be better despite power cost because the security benefit outweighs the small battery hit"

### 📈 **Multi-Objective Balancing**  
RL automatically learns to balance:
- Power efficiency vs security strength
- Current performance vs future-proofing
- Individual transaction cost vs long-term system health

### 🎯 **Context Awareness**
RL considers combinations that humans might miss:
- Battery level AND threat level AND mission type
- Historical performance patterns
- Long-term consequences of decisions

### 🚀 **Continuous Improvement**
As conditions change, RL adapts:
- New threat patterns → Adjust security preferences
- Better hardware → Can afford more complex algorithms  
- Changing mission profiles → Rebalance priorities

---

## ✅ Check Your Understanding

### Quick Quiz:
1. **Which algorithms are quantum-resistant?**
   - [ ] ASCON, SPECK, HIGHT, CAMELLIA
   - [ ] KYBER, DILITHIUM, SPHINCS+, FALCON
   - [ ] Only SPHINCS+ and FALCON
   - [ ] All except SPECK

2. **What's the main trade-off between algorithm categories?**
   - [ ] Speed vs accuracy
   - [ ] Security vs complexity  
   - [ ] Power efficiency vs quantum resistance
   - [ ] Size vs performance

3. **Which algorithm would RL likely choose for (Very Low Battery, High Threat, Critical)?**
   - [ ] SPHINCS+ (maximum security)
   - [ ] SPECK (maximum efficiency)
   - [ ] ASCON (balanced efficiency with decent security)
   - [ ] DILITHIUM (good post-quantum option)

**Answers**: 1-b, 2-c, 3-c

### Discussion Questions:
1. Why might an RL system sometimes choose a less secure algorithm?  
2. How will the algorithm landscape change as quantum computers develop?
3. What new factors might we need to consider in future versions?

---

## 🚀 Next Steps

Great! You now understand the cryptographic algorithms our system chooses between.

**Next Reading**: `why_battery_optimization.md` - Learn why battery optimization is crucial for cryptographic systems.

**Key Takeaway**: Our 8 algorithms provide a spectrum of trade-offs between power efficiency, security strength, and future-proofing. RL learns the optimal selection strategy for each situation.

---

## 📚 Additional Resources  

### Algorithm Specifications:
- **NIST Standards**: Official specifications for post-quantum algorithms
- **Lightweight Crypto**: ASCON and other efficient algorithms  
- **Comparative Studies**: Academic papers comparing algorithm performance

### Implementation Details:
- `../environment/crypto_environment.py` - See how algorithms are modeled
- `../data/algorithm_specifications.json` - Detailed algorithm parameters
- `../benchmarks/` - Performance measurement results

### Further Reading:
- "Post-Quantum Cryptography for Dummies" - Accessible introduction
- NIST Post-Quantum Standardization Process
- "Lightweight Cryptography: The Next Generation" - Research overview

**Happy Learning!** 🔐
