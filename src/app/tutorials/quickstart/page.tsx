'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import { useState } from 'react';

export default function Quickstart() {
  const [activeStep, setActiveStep] = useState(0);

  const tutorialSteps = [
    {
      title: 'Environment Setup',
      description: 'Create a simple environment for our agent to learn',
      timeEstimate: '2 minutes',
      code: `import gym
import numpy as np
from toolbrain import ToolBrain

# Create a simple custom environment
class SimpleEnvironment:
    def __init__(self):
        self.state = 0
        self.max_steps = 100
        self.current_step = 0
        
    def reset(self):
        self.state = np.random.randint(0, 10)
        self.current_step = 0
        return self.state
    
    def step(self, action):
        # Simple logic: move towards target (state 5)
        if action == 0:  # move left
            self.state = max(0, self.state - 1)
        elif action == 1:  # move right
            self.state = min(9, self.state + 1)
            
        # Calculate reward
        reward = -abs(self.state - 5)  # Closer to 5 = higher reward
        
        self.current_step += 1
        done = self.current_step >= self.max_steps or self.state == 5
        
        return self.state, reward, done, {}
    
    def get_observation_space(self):
        return 10  # 10 possible states
    
    def get_action_space(self):
        return 2   # 2 possible actions

# Initialize environment
env = SimpleEnvironment()
print("✅ Environment created successfully!")
print(f"Observation space: {env.get_observation_space()}")
print(f"Action space: {env.get_action_space()}")`
    },
    {
      title: 'Initialize ToolBrain',
      description: 'Set up ToolBrain with basic configuration',
      timeEstimate: '1 minute',
      code: `# Initialize ToolBrain
brain = ToolBrain()

# Configure for our environment
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'memory_size': 10000,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'target_update_freq': 100
}

brain.set_config(config)
print("✅ ToolBrain initialized with configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")`
    },
    {
      title: 'Create Agent',
      description: 'Create an intelligent agent using the ModelFactory',
      timeEstimate: '1 minute',
      code: `from toolbrain import ModelFactory

# Create model factory
factory = ModelFactory()

# Define model architecture for our simple environment
model_config = {
    'input_size': env.get_observation_space(),
    'hidden_layers': [64, 32],
    'output_size': env.get_action_space(),
    'activation': 'relu',
    'optimizer': 'adam'
}

# Create the agent model
agent_model = factory.create_model(
    model_type='dqn',  # Deep Q-Network
    config=model_config
)

print("✅ Agent model created successfully!")
print(f"Model architecture: {model_config['hidden_layers']}")
print(f"Input size: {model_config['input_size']}")
print(f"Output size: {model_config['output_size']}")`
    },
    {
      title: 'Setup Reward System',
      description: 'Configure the reward system for effective learning',
      timeEstimate: '2 minutes',
      code: `from toolbrain import RewardSystem

# Create reward system
reward_system = RewardSystem()

# Add reward components
reward_system.add_component(
    name='target_reaching',
    weight=1.0,
    description='Reward for reaching the target state'
)

reward_system.add_component(
    name='efficiency_bonus',
    weight=0.2,
    description='Bonus for reaching target quickly'
)

# Define custom reward shaping function
def shape_reward(state, action, next_state, base_reward, step_count):
    shaped_reward = base_reward
    
    # Add efficiency bonus
    if base_reward > -1:  # Close to target
        efficiency_bonus = max(0, (100 - step_count) / 100)
        shaped_reward += efficiency_bonus * 0.2
    
    # Add exploration bonus (small)
    exploration_bonus = 0.01
    shaped_reward += exploration_bonus
    
    return shaped_reward

reward_system.set_shaping_function(shape_reward)
print("✅ Reward system configured!")
print("Reward components:")
for component in reward_system.get_components():
    print(f"  - {component['name']}: weight={component['weight']}")`
    },
    {
      title: 'Training Loop',
      description: 'Train the agent with automatic optimization',
      timeEstimate: '5 minutes',
      code: `# Training configuration
training_config = {
    'episodes': 500,
    'max_steps_per_episode': 100,
    'save_frequency': 50,
    'log_frequency': 10
}

# Start training
print("🚀 Starting training...")
print(f"Episodes: {training_config['episodes']}")
print("=" * 50)

episode_rewards = []
episode_steps = []

for episode in range(training_config['episodes']):
    state = env.reset()
    total_reward = 0
    steps = 0
    
    while steps < training_config['max_steps_per_episode']:
        # Agent selects action
        action = brain.select_action(state, agent_model)
        
        # Environment step
        next_state, reward, done, _ = env.step(action)
        
        # Apply reward shaping
        shaped_reward = reward_system.shape_reward(
            state, action, next_state, reward, steps
        )
        
        # Store experience
        brain.store_experience(state, action, shaped_reward, next_state, done)
        
        # Train if enough experiences
        if brain.can_train():
            loss = brain.train_step(agent_model)
        
        total_reward += reward
        state = next_state
        steps += 1
        
        if done:
            break
    
    episode_rewards.append(total_reward)
    episode_steps.append(steps)
    
    # Log progress
    if episode % training_config['log_frequency'] == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        avg_steps = np.mean(episode_steps[-10:])
        print(f"Episode {episode}: Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}")
    
    # Update target network
    if episode % config['target_update_freq'] == 0:
        brain.update_target_network(agent_model)

print("✅ Training completed!")
print(f"Final average reward: {np.mean(episode_rewards[-50:]):.2f}")
print(f"Final average steps: {np.mean(episode_steps[-50:]):.1f}")`
    },
    {
      title: 'Test the Agent',
      description: 'Evaluate the trained agent performance',
      timeEstimate: '2 minutes',
      code: `# Test the trained agent
print("🧪 Testing trained agent...")
print("=" * 30)

test_episodes = 10
test_rewards = []
test_steps = []

# Disable exploration for testing
brain.set_testing_mode(True)

for episode in range(test_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    
    print(f"\\nTest Episode {episode + 1}:")
    print(f"Starting state: {state}")
    
    while steps < 100:
        # Agent selects best action (no exploration)
        action = brain.select_action(state, agent_model)
        next_state, reward, done, _ = env.step(action)
        
        print(f"Step {steps + 1}: State {state} -> Action {action} -> State {next_state} (Reward: {reward})")
        
        total_reward += reward
        state = next_state
        steps += 1
        
        if done:
            print(f"✅ Target reached in {steps} steps!")
            break
    
    test_rewards.append(total_reward)
    test_steps.append(steps)
    print(f"Episode reward: {total_reward}, Steps: {steps}")

print("\\n" + "=" * 50)
print("📊 Test Results:")
print(f"Average reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
print(f"Average steps: {np.mean(test_steps):.1f} ± {np.std(test_steps):.1f}")
print(f"Success rate: {sum(1 for r in test_rewards if r > -10) / len(test_rewards) * 100:.1f}%")

# Re-enable training mode
brain.set_testing_mode(False)`
    },
    {
      title: 'Save and Export',
      description: 'Save your trained model and configuration',
      timeEstimate: '1 minute',
      code: `# Save the trained model
model_path = "trained_agent.pth"
config_path = "agent_config.json"

# Save model
brain.save_model(agent_model, model_path)
print(f"✅ Model saved to: {model_path}")

# Save configuration
brain.save_config(config_path)
print(f"✅ Configuration saved to: {config_path}")

# Save training metrics
import json

metrics = {
    'training_episodes': len(episode_rewards),
    'final_avg_reward': float(np.mean(episode_rewards[-50:])),
    'final_avg_steps': float(np.mean(episode_steps[-50:])),
    'test_avg_reward': float(np.mean(test_rewards)),
    'test_avg_steps': float(np.mean(test_steps)),
    'success_rate': float(sum(1 for r in test_rewards if r > -10) / len(test_rewards))
}

with open('training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✅ Training metrics saved to: training_metrics.json")
print("\\n🎉 Quickstart tutorial completed successfully!")
print("\\nNext steps:")
print("- Try hyperparameter optimization with brain.optimize_hyperparameters()")
print("- Explore different model architectures")
print("- Learn about advanced reward shaping techniques")`
    }
  ];

  const progressPercentage = ((activeStep + 1) / tutorialSteps.length) * 100;

  return (
    <Layout>
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-[#E6EDF3] mb-4">Quick Start Tutorial</h1>
          <p className="text-xl text-gray-400 leading-relaxed mb-6">
            Build your first intelligent agent in under 15 minutes. This step-by-step tutorial 
            covers environment setup, agent creation, training, and evaluation.
          </p>
          
          {/* Progress Bar */}
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-[#E6EDF3]">Progress</span>
              <span className="text-sm text-gray-400">{Math.round(progressPercentage)}% complete</span>
            </div>
            <div className="w-full bg-[#21262D] rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-[#58A6FF] to-[#7C3AED] h-2 rounded-full transition-all duration-500"
                style={{ width: `${progressPercentage}%` }}
              ></div>
            </div>
            <div className="mt-2 text-sm text-gray-400">
              Step {activeStep + 1} of {tutorialSteps.length}: {tutorialSteps[activeStep].title}
            </div>
          </div>
        </div>

        {/* Step Navigation */}
        <div className="mb-8">
          <div className="flex flex-wrap gap-2">
            {tutorialSteps.map((step, index) => (
              <button
                key={index}
                onClick={() => setActiveStep(index)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 ${
                  index === activeStep
                    ? 'bg-[#58A6FF] text-white'
                    : index < activeStep
                    ? 'bg-[#3FB950] text-white'
                    : 'bg-[#161B22] text-gray-400 border border-[#30363D] hover:border-[#58A6FF]'
                }`}
              >
                {index < activeStep && '✅ '}
                Step {index + 1}
              </button>
            ))}
          </div>
        </div>

        {/* Current Step */}
        <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-8 mb-8">
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-2xl font-bold text-[#E6EDF3] mb-2">
                Step {activeStep + 1}: {tutorialSteps[activeStep].title}
              </h2>
              <p className="text-gray-400 mb-4">{tutorialSteps[activeStep].description}</p>
              <div className="flex items-center text-sm text-[#58A6FF]">
                <span className="mr-2">⏱️</span>
                Estimated time: {tutorialSteps[activeStep].timeEstimate}
              </div>
            </div>
          </div>
          
          <CodeBlock language="python">
            {tutorialSteps[activeStep].code}
          </CodeBlock>
        </div>

        {/* Navigation Buttons */}
        <div className="flex justify-between mb-8">
          <button
            onClick={() => setActiveStep(Math.max(0, activeStep - 1))}
            disabled={activeStep === 0}
            className={`px-6 py-3 rounded-lg font-medium transition-colors duration-200 ${
              activeStep === 0
                ? 'bg-[#21262D] text-gray-500 cursor-not-allowed'
                : 'bg-[#161B22] border border-[#30363D] text-[#E6EDF3] hover:border-[#58A6FF]'
            }`}
          >
            ← Previous Step
          </button>
          
          <button
            onClick={() => setActiveStep(Math.min(tutorialSteps.length - 1, activeStep + 1))}
            disabled={activeStep === tutorialSteps.length - 1}
            className={`px-6 py-3 rounded-lg font-medium transition-colors duration-200 ${
              activeStep === tutorialSteps.length - 1
                ? 'bg-[#21262D] text-gray-500 cursor-not-allowed'
                : 'bg-[#58A6FF] hover:bg-[#4A90E2] text-white'
            }`}
          >
            Next Step →
          </button>
        </div>

        {/* Additional Resources */}
        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
            <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">💡 Pro Tips</h3>
            <ul className="space-y-2 text-gray-400">
              <li>• Start with simple environments to understand the basics</li>
              <li>• Monitor training metrics to detect overfitting</li>
              <li>• Use reward shaping to guide agent behavior</li>
              <li>• Experiment with different model architectures</li>
              <li>• Save checkpoints regularly during long training sessions</li>
            </ul>
          </div>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
            <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">🚀 What's Next?</h3>
            <ul className="space-y-2 text-gray-400">
              <li>• <a href="/tutorials/hyperparameter-optimization" className="text-[#58A6FF] hover:underline">Hyperparameter Optimization Tutorial</a></li>
              <li>• <a href="/key-features/agent-training" className="text-[#58A6FF] hover:underline">Advanced Agent Training</a></li>
              <li>• <a href="/key-features/reward-system" className="text-[#58A6FF] hover:underline">Complex Reward Systems</a></li>
              <li>• <a href="/resources/examples" className="text-[#58A6FF] hover:underline">More Examples</a></li>
            </ul>
          </div>
        </div>

        {/* Completion Celebration */}
        {activeStep === tutorialSteps.length - 1 && (
          <div className="bg-gradient-to-r from-[#3FB950]/10 to-[#58A6FF]/10 border border-[#3FB950]/20 rounded-lg p-8 text-center">
            <div className="text-4xl mb-4">🎉</div>
            <h2 className="text-2xl font-bold text-[#E6EDF3] mb-4">Congratulations!</h2>
            <p className="text-gray-300 mb-6">
              You've successfully completed the ToolBrain quickstart tutorial! 
              You now have a trained intelligent agent and understand the core concepts.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-6 py-3 rounded-lg font-medium transition-colors duration-200">
                Explore Advanced Features
              </button>
              <button className="border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] px-6 py-3 rounded-lg font-medium transition-colors duration-200">
                Join Community
              </button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}