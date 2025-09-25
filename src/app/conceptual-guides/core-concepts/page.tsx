'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function CoreConcepts() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Core Concepts</h1>
          <p className="text-gray-300 text-lg">
            Understanding the fundamental concepts that make ToolBrain a powerful reinforcement learning platform.
          </p>
        </div>

        {/* Brain Architecture */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Brain Architecture</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              At the heart of ToolBrain is the <strong>Brain</strong> - an intelligent training orchestrator that manages the entire reinforcement learning workflow. The Brain automatically handles:
            </p>
            <ul className="list-disc list-inside text-gray-300 space-y-2 mb-6">
              <li>Model selection and architecture optimization</li>
              <li>Hyperparameter tuning using advanced algorithms</li>
              <li>Training progress monitoring and adaptive scheduling</li>
              <li>Performance evaluation and comparison</li>
              <li>Resource management and distributed training</li>
            </ul>
            
            <CodeBlock language="python">
{`from toolbrain import Brain

# Initialize the brain with your environment
brain = Brain(
    env_name="CartPole-v1",
    max_episodes=1000,
    optimization_budget=50
)

# The brain handles everything automatically
best_agent = brain.train()
results = brain.evaluate(best_agent)`}
            </CodeBlock>
          </div>
        </section>

        {/* Agent Lifecycle */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Agent Lifecycle</h2>
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">1. Initialization</h3>
              <p className="text-gray-300">
                Agents are created with default or specified configurations. The Brain analyzes the environment to suggest optimal starting parameters.
              </p>
            </div>
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">2. Training</h3>
              <p className="text-gray-300">
                Agents interact with the environment, collect experiences, and update their policies using reinforcement learning algorithms.
              </p>
            </div>
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">3. Evaluation</h3>
              <p className="text-gray-300">
                Performance is continuously monitored using multiple metrics. The Brain tracks progress and adjusts training dynamically.
              </p>
            </div>
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">4. Optimization</h3>
              <p className="text-gray-300">
                Hyperparameters are automatically tuned using Bayesian optimization and other advanced techniques to maximize performance.
              </p>
            </div>
          </div>
        </section>

        {/* Reward Engineering */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Reward Engineering</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              ToolBrain provides sophisticated reward engineering capabilities that go beyond simple reward functions:
            </p>
            
            <div className="mb-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Reward Shaping</h3>
              <p className="text-gray-300 mb-3">
                Automatically shapes rewards to guide learning toward desired behaviors while maintaining the optimal policy.
              </p>
              <CodeBlock language="python">
{`from toolbrain.rewards import RewardShaper

# Define a reward shaper with multiple objectives
shaper = RewardShaper(
    primary_reward=environment_reward,
    shaping_rewards={
        'progress': lambda state: distance_to_goal(state),
        'efficiency': lambda action: -energy_cost(action),
        'safety': lambda state: safety_margin(state)
    },
    weights={'progress': 0.3, 'efficiency': 0.2, 'safety': 0.5}
)

brain = Brain(env_name="MyEnv", reward_shaper=shaper)`}
              </CodeBlock>
            </div>

            <div className="mb-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Multi-Objective Optimization</h3>
              <p className="text-gray-300 mb-3">
                Balance multiple competing objectives using Pareto optimization and preference learning.
              </p>
              <CodeBlock language="python">
{`# Define multiple objectives
objectives = {
    'performance': lambda episode: episode.total_reward,
    'sample_efficiency': lambda episode: episode.total_reward / episode.steps,
    'robustness': lambda episode: episode.consistency_score
}

brain = Brain(
    env_name="Trading-v1",
    multi_objective=objectives,
    preference_learning=True
)`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* Model Factory */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Model Factory</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              The ModelFactory is ToolBrain's intelligent model selection and architecture optimization system:
            </p>
            
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold text-blue-400 mb-2">Algorithm Selection</h4>
                <p className="text-gray-300 text-sm">
                  Automatically chooses the best RL algorithm (PPO, SAC, TD3, etc.) based on environment characteristics.
                </p>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold text-blue-400 mb-2">Architecture Search</h4>
                <p className="text-gray-300 text-sm">
                  Neural Architecture Search (NAS) to find optimal network architectures for your specific problem.
                </p>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <h4 className="font-semibold text-blue-400 mb-2">Transfer Learning</h4>
                <p className="text-gray-300 text-sm">
                  Leverages pre-trained models and knowledge transfer to accelerate learning on new tasks.
                </p>
              </div>
            </div>

            <CodeBlock language="python">
{`from toolbrain.factory import ModelFactory

# Let the factory analyze your environment
factory = ModelFactory(env_name="AntBulletEnv-v0")

# Get recommendations based on environment analysis
recommendations = factory.analyze_environment()
print(f"Recommended algorithm: {recommendations.algorithm}")
print(f"Suggested architecture: {recommendations.architecture}")

# Create the optimal model automatically
model = factory.create_model(
    algorithm="auto",  # Let factory decide
    architecture="auto",  # Optimize architecture
    transfer_learning=True  # Use pre-trained knowledge
)`}
            </CodeBlock>
          </div>
        </section>

        {/* Training Strategies */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Training Strategies</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              ToolBrain employs advanced training strategies to maximize sample efficiency and final performance:
            </p>

            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">Curriculum Learning</h3>
                <p className="text-gray-300 mb-3">
                  Automatically generates learning curricula that start with simple tasks and progressively increase difficulty.
                </p>
                <CodeBlock language="python">
{`brain = Brain(
    env_name="NavigationMaze-v1",
    curriculum_learning=True,
    curriculum_strategy="progressive_difficulty"
)

# The brain automatically manages curriculum progression
# Starting with small mazes, moving to complex environments`}
                </CodeBlock>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">Population-Based Training</h3>
                <p className="text-gray-300 mb-3">
                  Trains multiple agents simultaneously with different hyperparameters, sharing knowledge between them.
                </p>
                <CodeBlock language="python">
{`brain = Brain(
    env_name="Trading-v1",
    population_size=10,  # Train 10 agents simultaneously
    population_strategy="exploit_and_explore",
    knowledge_sharing=True
)`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </section>

        {/* Performance Metrics */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Performance Metrics</h2>
          <div className="bg-gray-800 rounded-lg p-6">
            <p className="text-gray-300 mb-4">
              ToolBrain provides comprehensive performance tracking and analysis:
            </p>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">Training Metrics</h3>
                <ul className="list-disc list-inside text-gray-300 space-y-1">
                  <li>Episode rewards and returns</li>
                  <li>Sample efficiency curves</li>
                  <li>Policy gradient norms</li>
                  <li>Value function accuracy</li>
                  <li>Exploration vs exploitation balance</li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">Evaluation Metrics</h3>
                <ul className="list-disc list-inside text-gray-300 space-y-1">
                  <li>Success rate and consistency</li>
                  <li>Robustness to perturbations</li>
                  <li>Generalization capability</li>
                  <li>Resource utilization</li>
                  <li>Convergence stability</li>
                </ul>
              </div>
            </div>

            <div className="mt-6">
              <CodeBlock language="python">
{`# Access comprehensive metrics
results = brain.get_training_results()
print(f"Best episode reward: {results.best_reward}")
print(f"Sample efficiency: {results.sample_efficiency}")
print(f"Convergence time: {results.convergence_episodes}")

# Generate detailed performance report
report = brain.generate_report()
report.save("training_analysis.html")`}
              </CodeBlock>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}