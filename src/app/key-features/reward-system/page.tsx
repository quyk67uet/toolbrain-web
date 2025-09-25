'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import { useState } from 'react';

export default function RewardSystem() {
  const [selectedRewardType, setSelectedRewardType] = useState('sparse');

  const rewardTypes = [
    {
      id: 'sparse',
      name: 'Sparse Rewards',
      description: 'Rewards given only at specific milestones or task completion',
      icon: '🎯',
      advantages: ['Clear objective', 'No reward hacking', 'Simple to define'],
      challenges: ['Slow learning', 'Exploration difficulty', 'Credit assignment'],
      examples: ['Game completion', 'Goal reaching', 'Task success']
    },
    {
      id: 'dense',
      name: 'Dense Rewards',
      description: 'Frequent rewards provided at every step or action',
      icon: '🌊',
      advantages: ['Faster learning', 'Better guidance', 'Easier exploration'],
      challenges: ['Reward engineering', 'Local optima', 'Overfitting'],
      examples: ['Distance to goal', 'Progress metrics', 'Incremental improvements']
    },
    {
      id: 'shaped',
      name: 'Reward Shaping',
      description: 'Modified rewards to guide learning while preserving optimality',
      icon: '🔧',
      advantages: ['Preserved optimality', 'Faster convergence', 'Flexible design'],
      challenges: ['Complex design', 'Potential biasing', 'Domain knowledge needed'],
      examples: ['Potential-based shaping', 'Progress bonuses', 'Auxiliary rewards']
    },
    {
      id: 'intrinsic',
      name: 'Intrinsic Motivation',
      description: 'Self-generated rewards based on curiosity and exploration',
      icon: '🧠',
      advantages: ['No manual design', 'Exploration bonus', 'Transferable'],
      challenges: ['Complexity', 'Computational cost', 'Hyperparameter sensitivity'],
      examples: ['Curiosity-driven', 'Count-based', 'Information gain']
    }
  ];

  const rewardExamples = {
    sparse: `from toolbrain import ToolBrain, RewardSystem

# Create sparse reward system
reward_system = RewardSystem()

class SparseRewardEnvironment:
    def __init__(self, goal_position=(10, 10)):
        self.goal_position = goal_position
        self.agent_position = (0, 0)
        self.max_steps = 100
        self.current_step = 0
    
    def step(self, action):
        # Move agent based on action
        self.agent_position = self.update_position(action)
        self.current_step += 1
        
        # Sparse reward: only at goal or time limit
        reward = 0
        done = False
        
        if self.agent_position == self.goal_position:
            reward = 100  # Large positive reward for success
            done = True
        elif self.current_step >= self.max_steps:
            reward = -10  # Small negative reward for timeout
            done = True
        
        return self.agent_position, reward, done, {}
    
    def update_position(self, action):
        x, y = self.agent_position
        if action == 0:    # up
            y = min(10, y + 1)
        elif action == 1:  # down
            y = max(0, y - 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(10, x + 1)
        return (x, y)

# Configure agent for sparse rewards
brain = ToolBrain()
agent_config = {
    'algorithm': 'DQN',
    'exploration_strategy': 'epsilon_greedy',
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'replay_buffer_size': 100000,
    'learning_rate': 1e-3,
    # Important: Higher exploration for sparse rewards
    'exploration_bonus': 0.1
}

# Training with experience prioritization for sparse rewards
env = SparseRewardEnvironment()
agent = brain.create_agent(env.observation_space, env.action_space, agent_config)

def train_sparse_reward_agent():
    episodes = 2000  # More episodes needed for sparse rewards
    success_count = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store experience with priority for rare positive rewards
            priority = 1.0 if reward > 0 else 0.1
            agent.store_experience(state, action, reward, next_state, done, priority)
            
            total_reward += reward
            state = next_state
            
            if done:
                if reward > 0:  # Success
                    success_count += 1
                break
        
        # Train agent
        if agent.can_train():
            agent.train_step()
        
        # Logging
        if episode % 100 == 0:
            success_rate = success_count / (episode + 1)
            print(f"Episode {episode}: Success Rate = {success_rate:.2%}")

train_sparse_reward_agent()`,

    dense: `from toolbrain import ToolBrain, RewardSystem

class DenseRewardEnvironment:
    def __init__(self, goal_position=(10, 10)):
        self.goal_position = goal_position
        self.agent_position = (0, 0)
        self.previous_distance = self.calculate_distance()
        self.max_steps = 100
        self.current_step = 0
    
    def calculate_distance(self):
        x1, y1 = self.agent_position
        x2, y2 = self.goal_position
        return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    
    def step(self, action):
        old_position = self.agent_position
        self.agent_position = self.update_position(action)
        self.current_step += 1
        
        # Dense reward components
        current_distance = self.calculate_distance()
        
        # 1. Progress reward (negative of distance change)
        progress_reward = self.previous_distance - current_distance
        
        # 2. Goal completion bonus
        goal_reward = 100 if self.agent_position == self.goal_position else 0
        
        # 3. Step penalty (encourages efficiency)
        step_penalty = -0.1
        
        # 4. Wall collision penalty
        collision_penalty = -1 if self.hit_wall(old_position, self.agent_position) else 0
        
        # Combine all reward components
        total_reward = progress_reward + goal_reward + step_penalty + collision_penalty
        
        self.previous_distance = current_distance
        
        done = (self.agent_position == self.goal_position) or (self.current_step >= self.max_steps)
        
        return self.agent_position, total_reward, done, {
            'progress_reward': progress_reward,
            'goal_reward': goal_reward,
            'step_penalty': step_penalty,
            'collision_penalty': collision_penalty
        }
    
    def hit_wall(self, old_pos, new_pos):
        return old_pos == new_pos  # No movement means wall collision
    
    def update_position(self, action):
        x, y = self.agent_position
        if action == 0:    # up
            y = min(10, y + 1)
        elif action == 1:  # down
            y = max(0, y - 1)
        elif action == 2:  # left
            x = max(0, x - 1)
        elif action == 3:  # right
            x = min(10, x + 1)
        return (x, y)

# Configure agent for dense rewards
brain = ToolBrain()
agent_config = {
    'algorithm': 'PPO',
    'learning_rate': 3e-4,
    'batch_size': 64,
    'clip_ratio': 0.2,
    # Lower exploration needed with dense rewards
    'entropy_coefficient': 0.01
}

# Advanced dense reward system with automatic balancing
reward_system = RewardSystem()

# Add weighted reward components
reward_system.add_component(
    name='progress',
    weight=1.0,
    description='Reward for making progress toward goal'
)

reward_system.add_component(
    name='efficiency',
    weight=0.5,
    description='Penalty for taking too many steps'
)

reward_system.add_component(
    name='goal_completion',
    weight=10.0,
    description='Bonus for reaching the goal'
)

# Automatic reward scaling and normalization
def adaptive_dense_rewards(state, action, next_state, base_rewards, episode_stats):
    # Normalize rewards based on episode statistics
    progress_std = episode_stats.get('progress_std', 1.0)
    normalized_progress = base_rewards['progress_reward'] / max(progress_std, 0.1)
    
    # Adaptive step penalty based on current performance
    current_efficiency = episode_stats.get('avg_steps_to_goal', 100)
    adaptive_step_penalty = -0.1 * (current_efficiency / 50)  # Scale with efficiency
    
    # Combine normalized rewards
    final_reward = (
        normalized_progress * reward_system.get_weight('progress') +
        adaptive_step_penalty * reward_system.get_weight('efficiency') +
        base_rewards['goal_reward'] * reward_system.get_weight('goal_completion')
    )
    
    return final_reward

# Training with dense rewards
env = DenseRewardEnvironment()
agent = brain.create_agent(env.observation_space, env.action_space, agent_config)

episode_stats = {'progress_std': 1.0, 'avg_steps_to_goal': 100}

for episode in range(1000):
    state = env.reset()
    episode_rewards = []
    
    while True:
        action = agent.select_action(state)
        next_state, base_reward, done, info = env.step(action)
        
        # Apply adaptive reward shaping
        shaped_reward = adaptive_dense_rewards(state, action, next_state, info, episode_stats)
        
        agent.store_experience(state, action, shaped_reward, next_state, done)
        episode_rewards.append(info['progress_reward'])
        
        state = next_state
        if done:
            break
    
    # Update episode statistics for adaptive rewards
    episode_stats['progress_std'] = np.std(episode_rewards)
    
    if agent.can_train():
        agent.train_step()`,

    shaped: `from toolbrain import ToolBrain, RewardSystem
import numpy as np

class ShapedRewardSystem:
    """Potential-based reward shaping that preserves optimal policy"""
    
    def __init__(self, goal_position=(10, 10), gamma=0.99):
        self.goal_position = goal_position
        self.gamma = gamma
        self.previous_potential = None
    
    def potential_function(self, state):
        """Potential function based on distance to goal"""
        if state is None:
            return 0
        
        x, y = state
        goal_x, goal_y = self.goal_position
        distance = ((goal_x - x)**2 + (goal_y - y)**2)**0.5
        
        # Negative distance as potential (closer = higher potential)
        return -distance
    
    def shape_reward(self, state, action, next_state, original_reward):
        """Apply potential-based reward shaping"""
        current_potential = self.potential_function(state)
        next_potential = self.potential_function(next_state)
        
        # Shaping reward: γ * Φ(s') - Φ(s)
        shaping_reward = self.gamma * next_potential - current_potential
        
        # Combine with original reward
        shaped_reward = original_reward + shaping_reward
        
        return shaped_reward, {
            'original_reward': original_reward,
            'shaping_reward': shaping_reward,
            'current_potential': current_potential,
            'next_potential': next_potential
        }

# Advanced multi-component reward shaping
class MultiComponentRewardShaper:
    def __init__(self):
        self.components = {}
        self.weights = {}
        self.normalizers = {}
    
    def add_shaping_component(self, name, potential_fn, weight=1.0, normalize=True):
        """Add a shaping component with its potential function"""
        self.components[name] = potential_fn
        self.weights[name] = weight
        self.normalizers[name] = RunningMeanStd() if normalize else None
    
    def shape_reward(self, state, action, next_state, original_reward, gamma=0.99):
        """Apply multi-component reward shaping"""
        total_shaping = 0
        shaping_breakdown = {}
        
        for name, potential_fn in self.components.items():
            # Calculate potential difference
            current_potential = potential_fn(state)
            next_potential = potential_fn(next_state)
            component_shaping = gamma * next_potential - current_potential
            
            # Normalize if requested
            if self.normalizers[name] is not None:
                self.normalizers[name].update(component_shaping)
                component_shaping = self.normalizers[name].normalize(component_shaping)
            
            # Weight and accumulate
            weighted_shaping = component_shaping * self.weights[name]
            total_shaping += weighted_shaping
            shaping_breakdown[name] = weighted_shaping
        
        return original_reward + total_shaping, shaping_breakdown

# Example usage with complex environment
class ComplexNavigationEnvironment:
    def __init__(self):
        self.agent_pos = (0, 0)
        self.goal_pos = (10, 10)
        self.obstacles = [(3, 3), (3, 4), (4, 3), (7, 7), (7, 8), (8, 7)]
        self.treasures = [(2, 5), (6, 2), (8, 9)]
        self.collected_treasures = set()
        
    def step(self, action):
        old_pos = self.agent_pos
        self.agent_pos = self.move(action)
        
        # Base environment reward (sparse)
        reward = 0
        done = False
        
        # Goal reached
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
        
        # Treasure collection
        if self.agent_pos in self.treasures and self.agent_pos not in self.collected_treasures:
            self.collected_treasures.add(self.agent_pos)
            reward += 10
        
        # Obstacle collision
        if self.agent_pos in self.obstacles:
            reward -= 5
            self.agent_pos = old_pos  # Reset position
        
        return self.agent_pos, reward, done, {}
    
    def move(self, action):
        x, y = self.agent_pos
        if action == 0: y = min(10, y + 1)
        elif action == 1: y = max(0, y - 1)  
        elif action == 2: x = max(0, x - 1)
        elif action == 3: x = min(10, x + 1)
        return (x, y)

# Setup multi-component reward shaping
shaper = MultiComponentRewardShaper()

# Goal-seeking potential
def goal_potential(state):
    if state is None: return 0
    x, y = state
    dist = ((10 - x)**2 + (10 - y)**2)**0.5
    return -dist

# Obstacle avoidance potential  
def obstacle_potential(state):
    if state is None: return 0
    x, y = state
    min_obstacle_dist = min([((ox - x)**2 + (oy - y)**2)**0.5 
                            for ox, oy in [(3, 3), (3, 4), (4, 3), (7, 7), (7, 8), (8, 7)]])
    return min_obstacle_dist  # Higher potential when far from obstacles

# Treasure seeking potential
def treasure_potential(state, collected_treasures):
    if state is None: return 0
    x, y = state
    uncollected = [(2, 5), (6, 2), (8, 9)] - collected_treasures
    if not uncollected: return 0
    
    min_treasure_dist = min([((tx - x)**2 + (ty - y)**2)**0.5 for tx, ty in uncollected])
    return -min_treasure_dist

# Add shaping components
shaper.add_shaping_component('goal_seeking', goal_potential, weight=1.0)
shaper.add_shaping_component('obstacle_avoidance', obstacle_potential, weight=0.3)

# Custom training loop with shaped rewards
brain = ToolBrain()
env = ComplexNavigationEnvironment()
agent = brain.create_agent(env.observation_space, env.action_space, {
    'algorithm': 'A2C',
    'learning_rate': 1e-3
})

for episode in range(1000):
    state = env.reset()
    env.collected_treasures = set()  # Reset treasures
    
    while True:
        action = agent.select_action(state)
        next_state, original_reward, done, _ = env.step(action)
        
        # Apply reward shaping
        shaped_reward, shaping_breakdown = shaper.shape_reward(
            state, action, next_state, original_reward
        )
        
        agent.store_experience(state, action, shaped_reward, next_state, done)
        
        # Optional: Log shaping components for analysis
        if episode % 100 == 0:
            print(f"Original: {original_reward:.2f}, Shaped: {shaped_reward:.2f}")
            for component, value in shaping_breakdown.items():
                print(f"  {component}: {value:.3f}")
        
        state = next_state
        if done:
            break
    
    if agent.can_train():
        agent.train_step()`,

    intrinsic: `from toolbrain import ToolBrain, RewardSystem
import numpy as np
from collections import defaultdict

class CuriosityDrivenReward:
    """Intrinsic Curiosity Module (ICM) for exploration bonus"""
    
    def __init__(self, state_dim, action_dim, feature_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        
        # Neural networks for ICM
        self.feature_network = self.build_feature_network()
        self.forward_model = self.build_forward_model()
        self.inverse_model = self.build_inverse_model()
        
        self.curiosity_weight = 0.1
    
    def build_feature_network(self):
        """Learn feature representation of states"""
        # Simplified: In practice, use neural networks
        return lambda state: self.hash_state(state)
    
    def build_forward_model(self):
        """Predict next state features from current features + action"""
        # Forward model: φ(s_t), a_t -> φ(s_t+1)
        return lambda features, action: features + 0.1 * action  # Simplified
    
    def build_inverse_model(self):
        """Predict action from state transition"""
        # Inverse model: φ(s_t), φ(s_t+1) -> a_t
        return lambda feat1, feat2: np.argmax(feat2 - feat1)  # Simplified
    
    def hash_state(self, state):
        """Convert state to feature representation"""
        if isinstance(state, tuple):
            return np.array(state, dtype=np.float32)
        return np.array([state], dtype=np.float32)
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute curiosity-driven intrinsic reward"""
        # Get feature representations
        current_features = self.feature_network(state)
        next_features = self.feature_network(next_state)
        
        # Forward model prediction error
        predicted_next_features = self.forward_model(current_features, action)
        forward_error = np.mean((predicted_next_features - next_features)**2)
        
        # Intrinsic reward is prediction error (curiosity)
        intrinsic_reward = self.curiosity_weight * forward_error
        
        return intrinsic_reward, {
            'forward_error': forward_error,
            'current_features': current_features,
            'next_features': next_features,
            'predicted_features': predicted_next_features
        }

class CountBasedExploration:
    """Count-based exploration bonus"""
    
    def __init__(self, bonus_coefficient=0.1, decay_rate=0.99):
        self.state_counts = defaultdict(int)
        self.bonus_coefficient = bonus_coefficient
        self.decay_rate = decay_rate
        self.total_steps = 0
    
    def get_state_key(self, state):
        """Convert state to hashable key"""
        if isinstance(state, (tuple, list)):
            return tuple(state)
        return state
    
    def compute_exploration_bonus(self, state):
        """Compute exploration bonus based on visit count"""
        state_key = self.get_state_key(state)
        visit_count = self.state_counts[state_key]
        
        # Bonus inversely related to visit count
        if visit_count == 0:
            bonus = self.bonus_coefficient
        else:
            bonus = self.bonus_coefficient / np.sqrt(visit_count)
        
        # Update count
        self.state_counts[state_key] += 1
        self.total_steps += 1
        
        # Decay bonus over time to reduce exploration as training progresses
        decayed_bonus = bonus * (self.decay_rate ** (self.total_steps / 1000))
        
        return decayed_bonus, {
            'visit_count': visit_count + 1,
            'raw_bonus': bonus,
            'decayed_bonus': decayed_bonus
        }

class InformationGainReward:
    """Information gain based intrinsic motivation"""
    
    def __init__(self, history_size=1000):
        self.state_history = []
        self.history_size = history_size
        self.information_weight = 0.05
    
    def compute_information_gain(self, state):
        """Compute information gain from visiting this state"""
        state_array = np.array(state if isinstance(state, (list, tuple)) else [state])
        
        if len(self.state_history) < 2:
            information_gain = self.information_weight
        else:
            # Compute distance to most similar historical state
            distances = [np.linalg.norm(state_array - hist_state) 
                        for hist_state in self.state_history]
            min_distance = min(distances)
            
            # Information gain inversely related to similarity
            information_gain = self.information_weight * min_distance
        
        # Add to history
        self.state_history.append(state_array)
        if len(self.state_history) > self.history_size:
            self.state_history.pop(0)
        
        return information_gain

# Combined intrinsic motivation system
class IntrinsicMotivationSystem:
    def __init__(self, state_dim, action_dim):
        self.curiosity_module = CuriosityDrivenReward(state_dim, action_dim)
        self.count_explorer = CountBasedExploration()
        self.info_gain_module = InformationGainReward()
        
        # Weights for combining different intrinsic rewards
        self.curiosity_weight = 0.4
        self.count_weight = 0.3
        self.info_gain_weight = 0.3
    
    def compute_intrinsic_reward(self, state, action, next_state):
        """Combine multiple sources of intrinsic motivation"""
        
        # Curiosity-driven reward
        curiosity_reward, curiosity_info = self.curiosity_module.compute_intrinsic_reward(
            state, action, next_state
        )
        
        # Count-based exploration bonus
        count_bonus, count_info = self.count_explorer.compute_exploration_bonus(next_state)
        
        # Information gain reward
        info_gain_reward = self.info_gain_module.compute_information_gain(next_state)
        
        # Combine intrinsic rewards
        total_intrinsic = (
            self.curiosity_weight * curiosity_reward +
            self.count_weight * count_bonus + 
            self.info_gain_weight * info_gain_reward
        )
        
        return total_intrinsic, {
            'curiosity': curiosity_reward,
            'count_bonus': count_bonus,
            'info_gain': info_gain_reward,
            'total_intrinsic': total_intrinsic,
            'curiosity_info': curiosity_info,
            'count_info': count_info
        }

# Training with intrinsic motivation
brain = ToolBrain()
env = ComplexNavigationEnvironment()  # Use previous complex environment

# Initialize intrinsic motivation system
intrinsic_system = IntrinsicMotivationSystem(
    state_dim=2,  # (x, y) position
    action_dim=4  # 4 movement directions
)

agent = brain.create_agent(env.observation_space, env.action_space, {
    'algorithm': 'PPO',
    'learning_rate': 1e-3,
    'exploration_strategy': 'intrinsic_motivation'
})

def train_with_intrinsic_motivation():
    episodes = 1500
    
    for episode in range(episodes):
        state = env.reset()
        episode_extrinsic_reward = 0
        episode_intrinsic_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, extrinsic_reward, done, _ = env.step(action)
            
            # Compute intrinsic reward
            intrinsic_reward, intrinsic_info = intrinsic_system.compute_intrinsic_reward(
                state, action, next_state
            )
            
            # Combine extrinsic and intrinsic rewards
            total_reward = extrinsic_reward + intrinsic_reward
            
            agent.store_experience(state, action, total_reward, next_state, done)
            
            episode_extrinsic_reward += extrinsic_reward
            episode_intrinsic_reward += intrinsic_reward
            
            state = next_state
            if done:
                break
        
        # Train agent
        if agent.can_train():
            agent.train_step()
        
        # Logging
        if episode % 100 == 0:
            total_states_explored = len(intrinsic_system.count_explorer.state_counts)
            print(f"Episode {episode}:")
            print(f"  Extrinsic reward: {episode_extrinsic_reward:.2f}")
            print(f"  Intrinsic reward: {episode_intrinsic_reward:.2f}")
            print(f"  States explored: {total_states_explored}")

train_with_intrinsic_motivation()`
  };

  return (
    <Layout>
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-[#E6EDF3] mb-4">Reward System Design</h1>
          <p className="text-xl text-gray-400 leading-relaxed">
            Design effective reward systems that guide agent learning. Master sparse rewards, dense rewards, 
            reward shaping, and intrinsic motivation techniques for optimal training performance.
          </p>
        </div>

        {/* Reward Types Overview */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Reward Design Approaches</h2>
          
          <div className="grid md:grid-cols-2 gap-6 mb-8">
            {rewardTypes.map((type) => (
              <div
                key={type.id}
                onClick={() => setSelectedRewardType(type.id)}
                className={`cursor-pointer p-6 rounded-lg border transition-all duration-200 ${
                  selectedRewardType === type.id
                    ? 'bg-[#58A6FF]/10 border-[#58A6FF]'
                    : 'bg-[#161B22] border-[#30363D] hover:border-[#58A6FF]/50'
                }`}
              >
                <div className="flex items-center mb-4">
                  <div className="text-2xl mr-3">{type.icon}</div>
                  <h3 className="text-lg font-semibold text-[#E6EDF3]">{type.name}</h3>
                </div>
                
                <p className="text-gray-400 mb-4 text-sm">{type.description}</p>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <div className="text-xs font-medium text-[#3FB950] mb-2">✅ ADVANTAGES</div>
                    <ul className="text-xs text-gray-400 space-y-1">
                      {type.advantages.slice(0, 2).map((adv, index) => (
                        <li key={index}>• {adv}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div>
                    <div className="text-xs font-medium text-[#F85149] mb-2">❌ CHALLENGES</div>
                    <ul className="text-xs text-gray-400 space-y-1">
                      {type.challenges.slice(0, 2).map((ch, index) => (
                        <li key={index}>• {ch}</li>
                      ))}
                    </ul>
                  </div>
                </div>
                
                <div>
                  <div className="text-xs font-medium text-[#58A6FF] mb-2">🎯 EXAMPLES</div>
                  <div className="text-xs text-gray-400">
                    {type.examples.join(', ')}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Reward Type Details */}
        <div className="mb-12">
        {(() => {
          const selectedType = rewardTypes.find(t => t.id === selectedRewardType);
          if (!selectedType) return null;
          return (
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-8">
              <div className="flex items-center mb-6">
                <div className="text-3xl mr-4">{selectedType.icon}</div>
                <div>
                  <h2 className="text-2xl font-bold text-[#E6EDF3]">{selectedType.name}</h2>
                  <p className="text-gray-400">{selectedType.description}</p>
                </div>
              </div>                <div className="grid md:grid-cols-3 gap-6 mb-8">
                  <div className="bg-[#0D1117] rounded-lg p-4">
                    <h3 className="font-semibold text-[#3FB950] mb-3">✅ Advantages</h3>
                    <ul className="space-y-1 text-sm text-gray-400">
                      {selectedType.advantages.map((adv, index) => (
                        <li key={index}>• {adv}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="bg-[#0D1117] rounded-lg p-4">
                    <h3 className="font-semibold text-[#F85149] mb-3">❌ Challenges</h3>
                    <ul className="space-y-1 text-sm text-gray-400">
                      {selectedType.challenges.map((ch, index) => (
                        <li key={index}>• {ch}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="bg-[#0D1117] rounded-lg p-4">
                    <h3 className="font-semibold text-[#58A6FF] mb-3">🎯 Examples</h3>
                    <ul className="space-y-1 text-sm text-gray-400">
                      {selectedType.examples.map((ex, index) => (
                        <li key={index}>• {ex}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">Implementation Example</h3>
                <CodeBlock language="python">
                  {rewardExamples[selectedRewardType as keyof typeof rewardExamples]}
                </CodeBlock>
              </div>
            );
          })()}
        </div>

        {/* Reward Engineering Best Practices */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Reward Engineering Best Practices</h2>
          
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">🎯 Design Principles</h3>
                <ul className="space-y-3 text-gray-400">
                  <li className="flex items-start">
                    <span className="text-[#3FB950] mr-2">•</span>
                    <div>
                      <span className="font-medium text-[#E6EDF3]">Alignment:</span> Ensure rewards align with true objectives
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-[#3FB950] mr-2">•</span>
                    <div>
                      <span className="font-medium text-[#E6EDF3]">Consistency:</span> Maintain consistent reward signals across episodes
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-[#3FB950] mr-2">•</span>
                    <div>
                      <span className="font-medium text-[#E6EDF3]">Balance:</span> Balance exploration and exploitation rewards
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-[#3FB950] mr-2">•</span>
                    <div>
                      <span className="font-medium text-[#E6EDF3]">Scalability:</span> Design rewards that scale with problem complexity
                    </div>
                  </li>
                </ul>
              </div>
              
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">⚠️ Common Pitfalls</h3>
                <ul className="space-y-3 text-gray-400">
                  <li className="flex items-start">
                    <span className="text-[#F85149] mr-2">•</span>
                    <div>
                      <span className="font-medium text-[#E6EDF3]">Reward Hacking:</span> Agent exploits loopholes in reward design
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-[#F85149] mr-2">•</span>
                    <div>
                      <span className="font-medium text-[#E6EDF3]">Sparse Rewards:</span> Too little guidance leads to poor exploration
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-[#F85149] mr-2">•</span>
                    <div>
                      <span className="font-medium text-[#E6EDF3]">Dense Rewards:</span> Over-guidance can lead to local optima
                    </div>
                  </li>
                  <li className="flex items-start">
                    <span className="text-[#F85149] mr-2">•</span>
                    <div>
                      <span className="font-medium text-[#E6EDF3]">Scale Issues:</span> Poorly scaled rewards cause training instability
                    </div>
                  </li>
                </ul>
              </div>
            </div>
            
            <div className="space-y-6">
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">🔧 Debugging Rewards</h3>
                <CodeBlock language="python">
{`# Reward analysis and debugging tools
from toolbrain.analysis import RewardAnalyzer

analyzer = RewardAnalyzer()

# Analyze reward distribution
reward_stats = analyzer.analyze_rewards(episode_rewards)
print(f"Mean: {reward_stats['mean']:.3f}")
print(f"Std: {reward_stats['std']:.3f}")
print(f"Sparsity: {reward_stats['sparsity']:.2%}")

# Detect reward hacking
hacking_indicators = analyzer.detect_reward_hacking(
    agent_trajectories, reward_components
)

# Visualize reward landscape
analyzer.plot_reward_landscape(environment)`}
                </CodeBlock>
              </div>
              
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">📊 Reward Metrics</h3>
                <CodeBlock language="python">
{`# Key metrics for reward evaluation
metrics = {
    'reward_variance': np.var(episode_rewards),
    'reward_skewness': scipy.stats.skew(episode_rewards),
    'reward_entropy': calculate_entropy(reward_histogram),
    'convergence_rate': measure_convergence_speed(),
    'stability_index': calculate_stability(reward_trajectory)
}

# Track reward component contributions
component_analysis = analyzer.component_importance(
    reward_components=['task', 'efficiency', 'exploration']
)`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </div>

        {/* Advanced Reward Techniques */}
        <div className="bg-gradient-to-r from-[#58A6FF]/10 to-[#7C3AED]/10 border border-[#58A6FF]/20 rounded-lg p-8">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">🚀 Advanced Reward Techniques</h2>
          
          <div className="grid lg:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">🔄 Adaptive Reward Systems</h3>
              <p className="text-gray-300 mb-4">
                Automatically adjust reward components based on training progress and agent performance.
              </p>
              <CodeBlock language="python">
{`# Adaptive reward scaling
class AdaptiveRewardSystem:
    def __init__(self):
        self.performance_history = []
        self.reward_scales = {'exploration': 1.0, 'task': 1.0}
    
    def adapt_rewards(self, current_performance):
        # Increase exploration if performance stagnates
        if self.is_stagnating():
            self.reward_scales['exploration'] *= 1.1
        
        # Reduce exploration as performance improves
        if self.is_improving():
            self.reward_scales['exploration'] *= 0.95
        
        return self.reward_scales`}
              </CodeBlock>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">🎭 Hierarchical Rewards</h3>
              <p className="text-gray-300 mb-4">
                Structure rewards across multiple levels for complex, long-horizon tasks.
              </p>
              <CodeBlock language="python">
{`# Hierarchical reward structure
hierarchy = {
    'high_level': {
        'goal_completion': 100,
        'milestone_progress': 20
    },
    'mid_level': {
        'subgoal_achievement': 10,
        'skill_execution': 5
    },
    'low_level': {
        'action_efficiency': 0.1,
        'safety_compliance': 0.5
    }
}

total_reward = sum(
    sum(level_rewards.values()) 
    for level_rewards in hierarchy.values()
)`}
              </CodeBlock>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}