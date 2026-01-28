import numpy as np


class RewardCalculator:
    
    def __init__(self):
        self.hit_reward_head = 10.0
        self.hit_reward_body = 5.0
        self.guard_bonus = 0.1
        self.extension_bonus = 0.05
        self.energy_penalty = 0.01
        self.stability_penalty = 0.5
        self.hit_penalty_head = 8.0
        self.hit_penalty_body = 3.0
        
    def compute(self, state, action, next_state, info):
        reward = 0.0
        
        if info.get("landed_hit"):
            if info.get("hit_location") == "head":
                reward += self.hit_reward_head
            else:
                reward += self.hit_reward_body
        
        if info.get("received_hit"):
            if info.get("hit_location") == "head":
                reward -= self.hit_penalty_head
            else:
                reward -= self.hit_penalty_body
        
        guard_score = self._compute_guard_score(state)
        reward += self.guard_bonus * guard_score
        
        extension_score = self._compute_extension_score(state, info.get("opponent_pos"))
        reward += self.extension_bonus * extension_score
        
        energy = np.sum(np.square(action))
        reward -= self.energy_penalty * energy
        
        torso_pitch = state.get("torso_pitch", 0)
        torso_roll = state.get("torso_roll", 0)
        if abs(torso_pitch) > 0.3 or abs(torso_roll) > 0.3:
            reward -= self.stability_penalty
        
        return reward
    
    def _compute_guard_score(self, state):
        left_fist = state.get("left_fist_pos", np.zeros(3))
        right_fist = state.get("right_fist_pos", np.zeros(3))
        head_pos = state.get("head_pos", np.array([0, 0, 1.5]))
        
        left_dist = np.linalg.norm(left_fist - head_pos)
        right_dist = np.linalg.norm(right_fist - head_pos)
        
        score = 0.0
        if left_dist < 0.3:
            score += 0.5
        if right_dist < 0.3:
            score += 0.5
        
        return score
    
    def _compute_extension_score(self, state, opponent_pos):
        if opponent_pos is None:
            return 0.0
        
        left_fist = state.get("left_fist_pos", np.zeros(3))
        right_fist = state.get("right_fist_pos", np.zeros(3))
        base_pos = state.get("base_pos", np.zeros(3))
        
        opponent_pos = np.array(opponent_pos)
        
        left_to_opp = opponent_pos - left_fist
        right_to_opp = opponent_pos - right_fist
        base_to_opp = opponent_pos - base_pos
        
        base_dist = np.linalg.norm(base_to_opp)
        left_dist = np.linalg.norm(left_to_opp)
        right_dist = np.linalg.norm(right_to_opp)
        
        score = 0.0
        if left_dist < base_dist:
            score += (base_dist - left_dist)
        if right_dist < base_dist:
            score += (base_dist - right_dist)
        
        return score


class DenseRewardShaper:
    
    def __init__(self, base_calculator=None):
        self.calculator = base_calculator or RewardCalculator()
        self.prev_left_dist = None
        self.prev_right_dist = None
        
    def reset(self):
        self.prev_left_dist = None
        self.prev_right_dist = None
    
    def compute(self, state, action, next_state, info):
        base_reward = self.calculator.compute(state, action, next_state, info)
        
        opponent_pos = info.get("opponent_pos", np.array([1.5, 0, 1.0]))
        left_fist = next_state.get("left_fist_pos", np.zeros(3))
        right_fist = next_state.get("right_fist_pos", np.zeros(3))
        
        left_dist = np.linalg.norm(left_fist - opponent_pos)
        right_dist = np.linalg.norm(right_fist - opponent_pos)
        
        approach_bonus = 0.0
        if self.prev_left_dist is not None:
            if left_dist < self.prev_left_dist:
                approach_bonus += 0.1 * (self.prev_left_dist - left_dist)
            if right_dist < self.prev_right_dist:
                approach_bonus += 0.1 * (self.prev_right_dist - right_dist)
        
        self.prev_left_dist = left_dist
        self.prev_right_dist = right_dist
        
        return base_reward + approach_bonus
