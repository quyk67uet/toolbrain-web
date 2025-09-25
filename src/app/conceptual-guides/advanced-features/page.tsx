'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function AdvancedFeatures() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Advanced Features</h1>
          <p className="text-gray-300 text-lg">
            Explore advanced capabilities that make ToolBrain suitable for complex, real-world applications.
          </p>
        </div>

        {/* Distributed Training */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Distributed Training</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Scale your training across multiple machines and GPUs with ToolBrain's built-in distributed training capabilities.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-blue-400 mb-2">Data Parallelism</h3>
                <p className="text-gray-300 text-sm">
                  Distribute experience collection across multiple workers while maintaining synchronized learning.
                </p>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-blue-400 mb-2">Model Parallelism</h3>
                <p className="text-gray-300 text-sm">
                  Split large neural networks across multiple devices for training massive models.
                </p>
              </div>
            </div>

            <CodeBlock language="python">
{`from toolbrain import Brain
from toolbrain.distributed import DistributedConfig

# Configure distributed training
dist_config = DistributedConfig(
    num_workers=8,           # Number of environment workers
    num_gpus=4,             # GPUs to use for training
    strategy="data_parallel", # or "model_parallel"
    communication_backend="nccl"
)

brain = Brain(
    env_name="ComplexSimulation-v1",
    distributed_config=dist_config,
    batch_size=512  # Larger batches for distributed training
)

# Training automatically scales across all resources
best_agent = brain.train()`}
            </CodeBlock>

            <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-4 mt-4">
              <h4 className="font-semibold text-yellow-400 mb-2">📊 Performance Benefits</h4>
              <ul className="text-yellow-200 text-sm space-y-1">
                <li>• 8x faster experience collection with 8 workers</li>
                <li>• 4x training speedup with 4 GPUs</li>
                <li>• Reduced wall-clock time from hours to minutes</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Meta-Learning */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Meta-Learning</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              ToolBrain implements state-of-the-art meta-learning algorithms that enable agents to quickly adapt to new tasks.
            </p>

            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">Model-Agnostic Meta-Learning (MAML)</h3>
                <p className="text-gray-300 mb-3">
                  Learn initialization parameters that allow rapid adaptation to new tasks with minimal data.
                </p>
                <CodeBlock language="python">
{`from toolbrain.meta import MAMLBrain

# Train a meta-learner on a family of related tasks
task_family = [
    "Navigation-Easy-v1",
    "Navigation-Medium-v1", 
    "Navigation-Hard-v1"
]

meta_brain = MAMLBrain(
    task_family=task_family,
    inner_lr=0.01,      # Learning rate for task adaptation
    outer_lr=0.001,     # Learning rate for meta-updates
    adaptation_steps=5   # Steps for task adaptation
)

# Train the meta-learner
meta_model = meta_brain.train()

# Quickly adapt to a new task
new_task_agent = meta_model.adapt_to_task(
    "Navigation-Expert-v1",
    adaptation_episodes=10  # Only 10 episodes needed!
)`}
                </CodeBlock>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">Few-Shot Learning</h3>
                <p className="text-gray-300 mb-3">
                  Leverage few-shot learning techniques to solve new tasks with minimal experience.
                </p>
                <CodeBlock language="python">
{`from toolbrain.meta import FewShotLearner

# Create a few-shot learner
learner = FewShotLearner(
    base_algorithm="PPO",
    meta_algorithm="Reptile",
    support_episodes=5,    # Examples per task
    query_episodes=10      # Evaluation episodes
)

brain = Brain(
    env_name="TaskDistribution-v1",
    meta_learner=learner,
    curriculum_learning=True
)`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </section>

        {/* Multi-Agent Systems */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Multi-Agent Systems</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Train multiple agents that can cooperate, compete, or coexist in complex environments.
            </p>

            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold text-blue-400 mb-2">Cooperative</h4>
                <p className="text-gray-300 text-sm">
                  Agents work together toward shared objectives using communication and coordination.
                </p>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold text-blue-400 mb-2">Competitive</h4>
                <p className="text-gray-300 text-sm">
                  Agents compete against each other, leading to emergent strategies and robust policies.
                </p>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold text-blue-400 mb-2">Mixed-Motive</h4>
                <p className="text-gray-300 text-sm">
                  Complex scenarios with both cooperative and competitive elements.
                </p>
              </div>
            </div>

            <CodeBlock language="python">
{`from toolbrain.multiagent import MultiAgentBrain

# Define agent roles and relationships
agent_config = {
    'cooperators': {
        'count': 3,
        'algorithm': 'MAPPO',  # Multi-Agent PPO
        'communication': True,
        'shared_rewards': True
    },
    'competitors': {
        'count': 2, 
        'algorithm': 'MASAC',  # Multi-Agent SAC
        'self_play': True,
        'population_based': True
    }
}

multi_brain = MultiAgentBrain(
    env_name="MultiAgentTrading-v1",
    agent_config=agent_config,
    coordination_mechanism="centralized_critic"
)

# Train all agents simultaneously
trained_agents = multi_brain.train()`}
            </CodeBlock>
          </div>
        </section>

        {/* Hierarchical Reinforcement Learning */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Hierarchical Reinforcement Learning</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Solve complex, long-horizon tasks by learning hierarchical policies with multiple levels of abstraction.
            </p>

            <div className="mb-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Options Framework</h3>
              <p className="text-gray-300 mb-3">
                Learn temporal abstractions (options) that represent higher-level skills and behaviors.
              </p>
              <CodeBlock language="python">
{`from toolbrain.hierarchical import HierarchicalBrain

# Define skill hierarchy
skill_hierarchy = {
    'high_level': {
        'skills': ['navigate', 'manipulate', 'communicate'],
        'horizon': 100,  # Steps per high-level action
        'algorithm': 'PPO'
    },
    'low_level': {
        'primitives': ['move_forward', 'turn_left', 'turn_right', 'grasp'],
        'horizon': 1,
        'algorithm': 'SAC'
    }
}

hier_brain = HierarchicalBrain(
    env_name="RobotManipulation-v1",
    hierarchy=skill_hierarchy,
    skill_discovery=True,  # Automatically discover useful skills
    transfer_learning=True
)`}
              </CodeBlock>
            </div>

            <div className="mb-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Goal-Conditioned Learning</h3>
              <p className="text-gray-300 mb-3">
                Train agents to achieve diverse goals using Hindsight Experience Replay and goal sampling.
              </p>
              <CodeBlock language="python">
{`from toolbrain.goals import GoalConditionedBrain

goal_brain = GoalConditionedBrain(
    env_name="FetchReach-v1",
    goal_space="continuous",
    goal_sampling_strategy="her",  # Hindsight Experience Replay
    curriculum_goals=True,         # Start with easy goals
    intrinsic_motivation=True      # Curiosity-driven exploration
)

# Train with automatic goal curriculum
agent = goal_brain.train()

# Test on specific goals
success_rate = goal_brain.evaluate_on_goals([
    [0.5, 0.3, 0.2],  # Target position 1
    [0.1, 0.8, 0.4],  # Target position 2
    [0.9, 0.2, 0.6]   # Target position 3
])`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* Continual Learning */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Continual Learning</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Enable agents to learn continuously from new experiences without forgetting previous knowledge.
            </p>

            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">Catastrophic Forgetting Prevention</h3>
                <p className="text-gray-300 mb-3">
                  Use elastic weight consolidation and other techniques to preserve important knowledge.
                </p>
                <CodeBlock language="python">
{`from toolbrain.continual import ContinualBrain

continual_brain = ContinualBrain(
    initial_env="Task1-v1",
    forgetting_prevention="ewc",    # Elastic Weight Consolidation
    memory_buffer_size=10000,       # Experience replay buffer
    importance_weight=1000          # Strength of forgetting prevention
)

# Learn task sequence
task_sequence = ["Task1-v1", "Task2-v1", "Task3-v1", "Task4-v1"]
for task in task_sequence:
    print(f"Learning {task}...")
    continual_brain.learn_task(task, episodes=1000)
    
    # Evaluate on all previous tasks
    performance = continual_brain.evaluate_all_tasks()
    print(f"Backward transfer: {performance.backward_transfer}")
    print(f"Forward transfer: {performance.forward_transfer}")`}
                </CodeBlock>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">Lifelong Learning</h3>
                <p className="text-gray-300 mb-3">
                  Continuously adapt to changing environments and new task distributions.
                </p>
                <CodeBlock language="python">
{`# Setup lifelong learning with drift detection
lifelong_brain = ContinualBrain(
    env_name="DynamicEnvironment-v1",
    drift_detection=True,
    adaptation_strategy="progressive_networks",
    knowledge_consolidation="gradient_episodic_memory"
)

# Continuously learn as environment changes
lifelong_brain.start_lifelong_learning(
    max_episodes=float('inf'),  # Run indefinitely
    adaptation_threshold=0.1,   # Detect when performance drops
    consolidation_frequency=1000 # How often to consolidate
)`}
              </CodeBlock>
              </div>
            </div>
          </div>
        </section>

        {/* Safety and Robustness */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Safety and Robustness</h2>
          <div className="bg-gray-800 rounded-lg p-6">
            <p className="text-gray-300 mb-4">
              Built-in safety mechanisms and robustness guarantees for deploying RL agents in critical applications.
            </p>

            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">Constrained RL</h3>
                <p className="text-gray-300 mb-3">
                  Enforce safety constraints during training and deployment.
                </p>
                <CodeBlock language="python">
{`from toolbrain.safety import ConstrainedBrain

# Define safety constraints
constraints = {
    'collision_avoidance': lambda state: min_distance(state) > 0.5,
    'speed_limit': lambda action: action.velocity < 10.0,
    'energy_budget': lambda episode: episode.total_energy < 1000
}

safe_brain = ConstrainedBrain(
    env_name="AutonomousDriving-v1",
    constraints=constraints,
    constraint_threshold=0.95,  # 95% constraint satisfaction
    lagrangian_multiplier=True  # Use constrained optimization
)`}
                </CodeBlock>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">Adversarial Training</h3>
                <p className="text-gray-300 mb-3">
                  Train robust policies that perform well under adversarial conditions.
                </p>
                <CodeBlock language="python">
{`from toolbrain.robustness import AdversarialBrain

robust_brain = AdversarialBrain(
    env_name="RobotControl-v1",
    adversarial_training=True,
    perturbation_budget=0.1,    # Maximum perturbation strength
    attack_frequency=0.2,       # 20% of episodes have attacks
    defense_strategy="domain_randomization"
)`}
                </CodeBlock>
              </div>
            </div>

            <div className="bg-red-900/20 border border-red-600 rounded-lg p-4">
              <h4 className="font-semibold text-red-400 mb-2">🛡️ Safety Features</h4>
              <ul className="text-red-200 text-sm space-y-1">
                <li>• Formal verification support for critical components</li>
                <li>• Out-of-distribution detection during deployment</li>
                <li>• Graceful degradation under unexpected conditions</li>
                <li>• Human-in-the-loop intervention capabilities</li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}