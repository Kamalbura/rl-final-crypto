"""
Cryptographic Algorithm Selection Environment
===========================================

This module implements the RL environment for battery-optimized
cryptographic algorithm selection using the defined state space.

Author: RL Team
Date: September 4, 2025
"""

import numpy as np
import random
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path to import state_space
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from state_space import StateSpace, CryptoState, CryptoAlgorithm

@dataclass
class EnvironmentInfo:
    """Additional information returned by environment"""
    expert_action: CryptoAlgorithm
    power_consumption: float
    security_level: str
    decision_rationale: str

class CryptoEnvironment:
    """
    RL Environment for Cryptographic Algorithm Selection
    
    State Space: 30 states (5 battery × 3 threat × 2 mission)
    Action Space: 8 algorithms (4 pre-quantum + 4 post-quantum)
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize the environment"""
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        self.state_space = StateSpace()
        self.current_state: Optional[CryptoState] = None
        self.step_count = 0
        self.episode_length = 100  # Maximum steps per episode
        
        # Environment dynamics (how states change)
        self.battery_drain_rate = 0.02  # 2% per step average
        self.threat_evolution_prob = 0.1  # 10% chance threat level changes
        self.mission_change_prob = 0.05   # 5% chance mission criticality changes
        
        print("🏗️ Crypto Environment initialized")
        print(f"   State Space: {self.state_space.total_states} states")
        print(f"   Action Space: {self.state_space.total_actions} actions")
    
    def reset(self, specific_state: Optional[CryptoState] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state
        
        Args:
            specific_state: If provided, start from this state
            
        Returns:
            observation: State encoded as numpy array
            info: Additional information dictionary
        """
        if specific_state is not None:
            self.current_state = specific_state
        else:
            # Start with random state
            random_index = np.random.randint(0, 30)
            self.current_state = CryptoState.from_index(random_index)
        
        self.step_count = 0
        
        observation = self._state_to_observation(self.current_state)
        info = {
            'state_description': self.current_state.get_description(),
            'expert_action': self.state_space.get_expert_action(self.current_state).name
        }
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment
        
        Args:
            action: Selected algorithm index (0-7)
            
        Returns:
            observation: Next state
            reward: Reward for this action
            terminated: Episode ended due to completion
            truncated: Episode ended due to time limit
            info: Additional information
        """
        if self.current_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        if not 0 <= action < 8:
            raise ValueError(f"Action must be 0-7, got {action}")
        
        # Convert action to algorithm
        selected_algorithm = CryptoAlgorithm(action)
        
        # Calculate reward
        reward = self._calculate_reward(self.current_state, selected_algorithm)
        
        # Create info dict
        expert_action = self.state_space.get_expert_action(self.current_state)
        algo_info = self.state_space.get_algorithm_info(selected_algorithm)
        
        info = EnvironmentInfo(
            expert_action=expert_action,
            power_consumption=algo_info['power_w'],
            security_level=algo_info['security_level'],
            decision_rationale=self._get_decision_rationale(self.current_state, selected_algorithm)
        )
        
        # Transition to next state
        next_state = self._transition_state(self.current_state, selected_algorithm)
        self.current_state = next_state
        self.step_count += 1
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.episode_length
        
        observation = self._state_to_observation(self.current_state)
        
        return observation, reward, terminated, truncated, info.__dict__
    
    def _state_to_observation(self, state: CryptoState) -> np.ndarray:
        """Convert state to observation vector"""
        # One-hot encoding of state components
        obs = np.zeros(10)  # 5 + 3 + 2 = 10 dimensions
        
        # Battery level (one-hot)
        obs[state.battery_level.value] = 1.0
        
        # Threat status (one-hot)
        obs[5 + state.threat_status.value] = 1.0
        
        # Mission criticality (one-hot) 
        obs[8 + state.mission_criticality.value] = 1.0
        
        return obs
    
    def _calculate_reward(self, state: CryptoState, action: CryptoAlgorithm) -> float:
        """
        Calculate reward for state-action pair
        
        Reward Components (as specified):
        - Battery Efficiency (40%): Power vs available battery
        - Security Appropriateness (40%): Algorithm vs threat level
        - Expert Agreement (20%): Bonus for following lookup table
        """
        reward = 0.0
        
        # Get expert recommendation
        expert_action = self.state_space.get_expert_action(state)
        
        # Component 1: Battery Efficiency (40% weight)
        battery_reward = self._calculate_battery_reward(state, action)
        reward += 0.4 * battery_reward
        
        # Component 2: Security Appropriateness (40% weight)
        security_reward = self._calculate_security_reward(state, action)
        reward += 0.4 * security_reward
        
        # Component 3: Expert Agreement (20% weight)
        expert_reward = self._calculate_expert_reward(action, expert_action)
        reward += 0.2 * expert_reward
        
        # Add small random noise for exploration
        reward += np.random.normal(0, 0.01)
        
        return reward
    
    def _calculate_battery_reward(self, state: CryptoState, action: CryptoAlgorithm) -> float:
        """Calculate battery efficiency reward"""
        power_consumption = self.state_space.algorithm_power[action]
        
        # Reward based on battery level appropriateness
        if state.battery_level == 0:  # Critical battery (<20%)
            if power_consumption <= 6.2:  # Efficient choice
                return 5.0
            else:  # High power when critical
                return -3.0
        
        elif state.battery_level == 1:  # Low battery (20-40%)
            if power_consumption <= 6.5:  # Reasonable choice
                return 3.0
            else:  # Too high power
                return -1.0
        
        elif state.battery_level == 2:  # Medium battery (40-60%)
            if 6.2 <= power_consumption <= 6.8:  # Good range
                return 4.0
            elif power_consumption < 6.2:  # Too conservative
                return 1.0
            else:  # Acceptable high power
                return 2.0
        
        elif state.battery_level == 3:  # Good battery (60-80%)
            if power_consumption >= 6.5:  # Can afford higher power
                return 4.0
            else:  # Conservative but acceptable
                return 2.0
        
        else:  # High battery (80-100%)
            if power_consumption >= 6.8:  # Use available power
                return 5.0
            else:  # Underutilizing power
                return 1.0
    
    def _calculate_security_reward(self, state: CryptoState, action: CryptoAlgorithm) -> float:
        """Calculate security appropriateness reward"""
        
        # Post-quantum algorithms preferred when threats present
        is_post_quantum = action.value >= 4
        
        if state.threat_status == 2:  # Confirmed threat
            if action == CryptoAlgorithm.FALCON:  # Highest security
                return 5.0
            elif is_post_quantum:  # Other PQC algorithms
                return 3.0
            else:  # Pre-quantum with confirmed threat
                return -5.0
        
        elif state.threat_status == 1:  # Confirming threat
            if is_post_quantum:  # Post-quantum appropriate
                if action == CryptoAlgorithm.FALCON:
                    return 4.0
                else:
                    return 3.0
            else:  # Pre-quantum with potential threat
                return -2.0
        
        else:  # Normal threat level
            if is_post_quantum:  # Post-quantum is safe choice
                return 2.0
            else:  # Pre-quantum acceptable when no threat
                if state.battery_level == 0:  # Critical battery exception
                    return 3.0
                else:  # Not ideal but acceptable
                    return 0.0
    
    def _calculate_expert_reward(self, action: CryptoAlgorithm, expert_action: CryptoAlgorithm) -> float:
        """Calculate expert agreement reward"""
        if action == expert_action:
            return 5.0  # Perfect match
        else:
            # Partial credit for reasonable alternatives
            action_power = self.state_space.algorithm_power[action]
            expert_power = self.state_space.algorithm_power[expert_action]
            
            power_diff = abs(action_power - expert_power)
            if power_diff <= 0.3:  # Very close power consumption
                return 2.0
            elif power_diff <= 0.6:  # Somewhat close
                return 1.0
            else:  # Far from expert choice
                return -1.0
    
    def _transition_state(self, current_state: CryptoState, action: CryptoAlgorithm) -> CryptoState:
        """Simulate state transition based on action"""
        new_battery = current_state.battery_level
        new_threat = current_state.threat_status  
        new_mission = current_state.mission_criticality
        
        # Battery drain based on power consumption
        power_consumption = self.state_space.algorithm_power[action]
        drain_probability = min(0.3, power_consumption / 20.0)  # Higher power = more drain
        
        if np.random.random() < drain_probability and new_battery.value > 0:
            new_battery = current_state.battery_level.__class__(new_battery.value - 1)
        
        # Threat evolution
        if np.random.random() < self.threat_evolution_prob:
            if new_threat.value < 2:  # Can escalate
                new_threat = current_state.threat_status.__class__(new_threat.value + 1)
            elif np.random.random() < 0.3:  # Can de-escalate
                new_threat = current_state.threat_status.__class__(max(0, new_threat.value - 1))
        
        # Mission criticality changes rarely
        if np.random.random() < self.mission_change_prob:
            new_mission = current_state.mission_criticality.__class__(1 - new_mission.value)
        
        return CryptoState(new_battery, new_threat, new_mission)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if battery is critical and threat is confirmed (emergency)
        return (self.current_state.battery_level == 0 and 
                self.current_state.threat_status == 2)
    
    def _get_decision_rationale(self, state: CryptoState, action: CryptoAlgorithm) -> str:
        """Generate human-readable decision rationale"""
        expert_action = self.state_space.get_expert_action(state)
        power = self.state_space.algorithm_power[action]
        
        if action == expert_action:
            return f"Follows expert recommendation: {action.name} ({power:.1f}W)"
        else:
            expert_power = self.state_space.algorithm_power[expert_action]
            return f"Deviates from expert: chose {action.name} ({power:.1f}W) vs expert {expert_action.name} ({expert_power:.1f}W)"
    
    def render(self, mode='human'):
        """Render current state"""
        if self.current_state is None:
            print("Environment not initialized")
            return
        
        print(f"\n{'='*50}")
        print(f"🔋 CRYPTO ENVIRONMENT STATE")
        print(f"{'='*50}")
        print(f"Step: {self.step_count}")
        print(f"State: {self.current_state.get_description()}")
        
        expert_action = self.state_space.get_expert_action(self.current_state)
        expert_power = self.state_space.algorithm_power[expert_action]
        print(f"Expert Recommendation: {expert_action.name} ({expert_power:.1f}W)")
        
        print(f"{'='*50}\n")
    
    def get_state_index(self) -> int:
        """Get current state as index"""
        return self.current_state.to_index() if self.current_state else -1

if __name__ == "__main__":
    # Test the environment
    print("🚀 Testing Crypto Environment")
    
    env = CryptoEnvironment(random_seed=42)
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial state: {info['state_description']}")
    print(f"Expert action: {info['expert_action']}")
    print(f"Observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(3):
        env.render()
        
        # Take random action
        action = np.random.randint(0, 8)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Action: {CryptoAlgorithm(action).name}")
        print(f"Reward: {reward:.3f}")
        print(f"Power: {info['power_consumption']:.1f}W")
        print(f"Rationale: {info['decision_rationale']}")
        
        if terminated or truncated:
            print("Episode ended!")
            break
    
    print("\n✅ Environment test completed!")
