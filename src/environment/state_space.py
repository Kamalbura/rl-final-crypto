"""
State Space Definition for Battery-Optimized Cryptographic RL
=============================================================

This module defines the complete state space encoding/decoding system
for the 30-state battery-optimized cryptographic algorithm selection.

Author: RL Team
Date: September 4, 2025
"""

import json
import numpy as np
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import IntEnum

class BatteryLevel(IntEnum):
    """Battery level discrete states"""
    CRITICAL = 0    # <20%
    LOW = 1         # 20-40%
    MEDIUM = 2      # 40-60%
    GOOD = 3        # 60-80%
    HIGH = 4        # 80-100%

class ThreatStatus(IntEnum):
    """Threat detection states"""
    NORMAL = 0      # No threat detected
    CONFIRMING = 1  # Potential threat being verified
    CONFIRMED = 2   # Threat confirmed

class MissionCriticality(IntEnum):
    """Mission importance levels"""
    ROUTINE = 0     # Standard operations
    IMPORTANT = 1   # Critical mission

class CryptoAlgorithm(IntEnum):
    """Available cryptographic algorithms"""
    # Pre-quantum algorithms (for emergency use)
    ASCON = 0
    SPECK = 1
    HIGHT = 2
    CAMELLIA = 3
    
    # Post-quantum algorithms (ordered by power consumption)
    KYBER = 4       # 6.2W - Lowest power PQC
    DILITHIUM = 5   # 6.5W - Medium power
    SPHINCS = 6     # 6.8W - High power
    FALCON = 7      # 7.1W - Highest power PQC

@dataclass
class CryptoState:
    """Represents a complete system state"""
    battery_level: BatteryLevel
    threat_status: ThreatStatus
    mission_criticality: MissionCriticality
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to tuple for hashing/indexing"""
        return (self.battery_level.value, self.threat_status.value, self.mission_criticality.value)
    
    def to_index(self) -> int:
        """Convert to single state index (0-29)"""
        return (self.battery_level.value * 6 + 
                self.threat_status.value * 2 + 
                self.mission_criticality.value)
    
    @classmethod
    def from_index(cls, index: int) -> 'CryptoState':
        """Create state from index (0-29)"""
        if not 0 <= index < 30:
            raise ValueError(f"State index must be 0-29, got {index}")
        
        mission = index % 2
        index //= 2
        threat = index % 3
        index //= 3
        battery = index
        
        return cls(
            battery_level=BatteryLevel(battery),
            threat_status=ThreatStatus(threat),
            mission_criticality=MissionCriticality(mission)
        )
    
    @classmethod
    def from_raw_values(cls, battery_percent: float, threat_level: int, is_important: bool) -> 'CryptoState':
        """Create state from raw sensor values"""
        # Discretize battery percentage
        if battery_percent < 20:
            battery = BatteryLevel.CRITICAL
        elif battery_percent < 40:
            battery = BatteryLevel.LOW
        elif battery_percent < 60:
            battery = BatteryLevel.MEDIUM
        elif battery_percent < 80:
            battery = BatteryLevel.GOOD
        else:
            battery = BatteryLevel.HIGH
        
        return cls(
            battery_level=battery,
            threat_status=ThreatStatus(threat_level),
            mission_criticality=MissionCriticality(int(is_important))
        )
    
    def get_description(self) -> str:
        """Human-readable description"""
        battery_desc = {
            BatteryLevel.CRITICAL: "<20%",
            BatteryLevel.LOW: "20-40%", 
            BatteryLevel.MEDIUM: "40-60%",
            BatteryLevel.GOOD: "60-80%",
            BatteryLevel.HIGH: "80-100%"
        }
        
        threat_desc = {
            ThreatStatus.NORMAL: "Normal",
            ThreatStatus.CONFIRMING: "Confirming", 
            ThreatStatus.CONFIRMED: "Confirmed"
        }
        
        mission_desc = {
            MissionCriticality.ROUTINE: "Routine",
            MissionCriticality.IMPORTANT: "Important"
        }
        
        return f"Battery: {battery_desc[self.battery_level]}, Threat: {threat_desc[self.threat_status]}, Mission: {mission_desc[self.mission_criticality]}"

class StateSpace:
    """Complete state space management"""
    
    def __init__(self):
        """Initialize with lookup table from JSON"""
        self.load_lookup_table()
        self.total_states = 30
        self.total_actions = 8
        
        # Create all possible states
        self.all_states = [CryptoState.from_index(i) for i in range(30)]
        
        # Algorithm power consumption (Watts)
        self.algorithm_power = {
            CryptoAlgorithm.ASCON: 5.8,
            CryptoAlgorithm.SPECK: 5.8,
            CryptoAlgorithm.HIGHT: 5.8,
            CryptoAlgorithm.CAMELLIA: 6.1,
            CryptoAlgorithm.KYBER: 6.2,
            CryptoAlgorithm.DILITHIUM: 6.5,
            CryptoAlgorithm.SPHINCS: 6.8,
            CryptoAlgorithm.FALCON: 7.1
        }
        
        # Pre-quantum selection weights for RANDOM_PRE
        self.prequantum_weights = {
            CryptoAlgorithm.HIGHT: 0.35,    # Fastest
            CryptoAlgorithm.SPECK: 0.35,    # Second fastest
            CryptoAlgorithm.ASCON: 0.25,    # Third fastest
            CryptoAlgorithm.CAMELLIA: 0.05  # Slowest - use sparingly
        }
    
    def load_lookup_table(self):
        """Load the expert lookup table from JSON"""
        try:
            with open('c:/Users/burak/Desktop/rl-final-crypto/data/state_mappings.json', 'r') as f:
                data = json.load(f)
            
            self.lookup_table = {}
            for entry in data['lookup_table']:
                state_tuple = (entry['Battery_Level'], entry['Threat_Status'], entry['Mission_Criticality'])
                algorithm = entry['Algorithm']
                self.lookup_table[state_tuple] = algorithm
            
            print("✅ Lookup table loaded successfully")
            
        except FileNotFoundError:
            print("⚠️ Lookup table not found, using default expert rules")
            self.lookup_table = self._create_default_lookup_table()
    
    def _create_default_lookup_table(self) -> Dict[Tuple[int, int, int], str]:
        """Create default lookup table if file not found"""
        table = {}
        
        for battery in range(5):
            for threat in range(3):
                for mission in range(2):
                    state_tuple = (battery, threat, mission)
                    
                    # Apply expert rules
                    if threat == 2:  # Confirmed threat
                        table[state_tuple] = "FALCON"
                    elif battery == 0 and threat == 0:  # Critical battery, no threat
                        table[state_tuple] = "RANDOM_PRE"
                    elif battery == 0:  # Critical battery with threat
                        table[state_tuple] = "KYBER"
                    elif battery == 1:  # Low battery
                        if threat >= 1 and mission == 1:
                            table[state_tuple] = "DILITHIUM"
                        else:
                            table[state_tuple] = "KYBER"
                    elif battery == 2:  # Medium battery
                        if threat >= 1 and mission == 1:
                            table[state_tuple] = "SPHINCS"
                        else:
                            table[state_tuple] = "DILITHIUM"
                    elif battery == 3:  # Good battery
                        if threat >= 1 and mission == 1:
                            table[state_tuple] = "FALCON"
                        else:
                            table[state_tuple] = "SPHINCS"
                    else:  # High battery
                        table[state_tuple] = "FALCON"
        
        return table
    
    def get_expert_action(self, state: CryptoState) -> CryptoAlgorithm:
        """Get expert recommendation for given state"""
        state_tuple = state.to_tuple()
        algorithm_name = self.lookup_table.get(state_tuple, "FALCON")
        
        # Handle RANDOM_PRE selection
        if algorithm_name == "RANDOM_PRE":
            # Use weighted random selection from pre-quantum algorithms
            algorithms = list(self.prequantum_weights.keys())
            weights = list(self.prequantum_weights.values())
            selected_idx = np.random.choice(len(algorithms), p=weights)
            return algorithms[selected_idx]
        
        # Map string names to enum values
        algorithm_map = {
            "KYBER": CryptoAlgorithm.KYBER,
            "DILITHIUM": CryptoAlgorithm.DILITHIUM,
            "SPHINCS": CryptoAlgorithm.SPHINCS,
            "FALCON": CryptoAlgorithm.FALCON,
            "ASCON": CryptoAlgorithm.ASCON,
            "SPECK": CryptoAlgorithm.SPECK,
            "HIGHT": CryptoAlgorithm.HIGHT,
            "CAMELLIA": CryptoAlgorithm.CAMELLIA
        }
        
        return algorithm_map.get(algorithm_name, CryptoAlgorithm.FALCON)
    
    def get_algorithm_info(self, algorithm: CryptoAlgorithm) -> Dict[str, Any]:
        """Get detailed information about an algorithm"""
        info = {
            "name": algorithm.name,
            "index": algorithm.value,
            "power_w": self.algorithm_power[algorithm],
            "type": "Pre-quantum" if algorithm.value < 4 else "Post-quantum",
            "security_level": "Low" if algorithm.value < 4 else "High"
        }
        
        if algorithm.value >= 4:  # Post-quantum
            power_rank = ["KYBER", "DILITHIUM", "SPHINCS", "FALCON"]
            info["pqc_power_rank"] = power_rank.index(algorithm.name) + 1
        
        return info
    
    def validate_state_space(self) -> bool:
        """Validate the complete state space"""
        print("🔍 Validating State Space...")
        
        # Check all states can be created and converted
        for i in range(30):
            state = CryptoState.from_index(i)
            assert state.to_index() == i, f"State {i} conversion failed"
            
            # Check expert action exists
            action = self.get_expert_action(state)
            assert isinstance(action, CryptoAlgorithm), f"Invalid action for state {i}"
        
        print("✅ State space validation passed")
        return True
    
    def print_summary(self):
        """Print complete state space summary"""
        print("\n" + "="*60)
        print("🏗️  BATTERY-OPTIMIZED CRYPTO RL - STATE SPACE SUMMARY")
        print("="*60)
        
        print(f"📊 Total States: {self.total_states}")
        print(f"🎮 Total Actions: {self.total_actions}")
        print(f"🔋 Battery Levels: {len(BatteryLevel)}")
        print(f"⚠️  Threat Levels: {len(ThreatStatus)}")
        print(f"📋 Mission Types: {len(MissionCriticality)}")
        
        print("\n🔌 Algorithm Power Consumption:")
        for algo in CryptoAlgorithm:
            info = self.get_algorithm_info(algo)
            print(f"  {info['name']:>12}: {info['power_w']:>4.1f}W ({info['type']})")
        
        print(f"\n📋 Lookup Table Entries: {len(self.lookup_table)}")
        print("="*60)
    
    def encode_state_for_dqn(self, state_index: int) -> np.ndarray:
        """
        Encode state index as feature vector for Deep Q-Learning
        
        Creates a 10-dimensional feature vector:
        - Battery level (5 dimensions: one-hot encoding)
        - Threat level (3 dimensions: one-hot encoding)  
        - Mission criticality (2 dimensions: one-hot encoding)
        
        Args:
            state_index: State index (0-29)
            
        Returns:
            10-dimensional numpy array representing the state
        """
        if not 0 <= state_index < 30:
            raise ValueError(f"State index {state_index} out of range [0, 29]")
        
        # Decode state index to components
        state = CryptoState.from_index(state_index)
        
        # Create feature vector
        feature_vector = np.zeros(10, dtype=np.float32)
        
        # Battery level (5 dimensions: one-hot)
        feature_vector[state.battery_level] = 1.0
        
        # Threat level (3 dimensions: one-hot, offset by 5)
        feature_vector[5 + state.threat_status] = 1.0
        
        # Mission criticality (2 dimensions: one-hot, offset by 8)
        feature_vector[8 + state.mission_criticality] = 1.0
        
        return feature_vector
    
    def encode_state_batch_for_dqn(self, state_indices: List[int]) -> np.ndarray:
        """
        Encode batch of state indices for DQN training
        
        Args:
            state_indices: List of state indices
            
        Returns:
            Numpy array of shape (batch_size, 10)
        """
        return np.array([self.encode_state_for_dqn(idx) for idx in state_indices])

if __name__ == "__main__":
    # Test the state space system
    print("🚀 Testing Battery-Optimized Crypto RL State Space")
    
    state_space = StateSpace()
    
    # Validate everything works
    state_space.validate_state_space()
    
    # Print summary
    state_space.print_summary()
    
    # Test a few example states
    print("\n🔍 Example State Tests:")
    
    # Critical battery, confirmed threat
    test_state = CryptoState.from_raw_values(15.0, 2, False)
    expert_action = state_space.get_expert_action(test_state)
    print(f"State: {test_state.get_description()}")
    print(f"Expert Action: {expert_action.name} ({state_space.algorithm_power[expert_action]:.1f}W)")
    
    # High battery, normal threat, important mission
    test_state = CryptoState.from_raw_values(90.0, 0, True)
    expert_action = state_space.get_expert_action(test_state)
    print(f"State: {test_state.get_description()}")
    print(f"Expert Action: {expert_action.name} ({state_space.algorithm_power[expert_action]:.1f}W)")
    
    print("\n✅ State Space System Ready!")
