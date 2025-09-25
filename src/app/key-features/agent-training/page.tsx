'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import { useState } from 'react';

export default function AgentTraining() {
  const [selectedTrainingType, setSelectedTrainingType] = useState('reinforcement');

  const trainingTypes = [
    {
      id: 'reinforcement',
      name: 'Reinforcement Learning',
      description: 'Learn through interaction with environment and reward signals',
      icon: '🎯',
      algorithms: ['DQN', 'PPO', 'SAC', 'A3C', 'DDPG'],
      useCases: ['Game playing', 'Robotics', 'Resource allocation', 'Trading']
    },
    {
      id: 'supervised',
      name: 'Supervised Learning',
      description: 'Learn from labeled training examples',
      icon: '📚',
      algorithms: ['Neural Networks', 'Decision Trees', 'SVM', 'Random Forest'],
      useCases: ['Classification', 'Regression', 'Pattern recognition', 'Prediction']
    },
    {
      id: 'unsupervised',
      name: 'Unsupervised Learning',
      description: 'Discover patterns in unlabeled data',
      icon: '🔍',
      algorithms: ['K-Means', 'PCA', 'Autoencoders', 'DBSCAN'],
      useCases: ['Clustering', 'Anomaly detection', 'Dimensionality reduction']
    },
    {
      id: 'meta',
      name: 'Meta Learning',
      description: 'Learn to learn new tasks quickly',
      icon: '🧠',
      algorithms: ['MAML', 'Prototypical Networks', 'Reptile', 'Meta-SGD'],
      useCases: ['Few-shot learning', 'Transfer learning', 'Adaptation']
    }
  ];

  const trainingExamples = {
    reinforcement: `from toolbrain import ToolBrain
from toolbrain.agents import RLAgent
from toolbrain.environments import create_environment

# Create reinforcement learning setup
brain = ToolBrain()
env = create_environment('CartPole-v1')

# Configure RL agent
rl_config = {
    'algorithm': 'PPO',  # Proximal Policy Optimization
    'policy_network': {
        'hidden_layers': [128, 128],
        'activation': 'tanh',
        'output_activation': 'softmax'
    },
    'value_network': {
        'hidden_layers': [128, 128],
        'activation': 'tanh'
    },
    'learning_rate': 3e-4,
    'clip_ratio': 0.2,
    'entropy_coefficient': 0.01,
    'value_coefficient': 0.5,
    'max_grad_norm': 0.5
}

# Create RL agent
agent = RLAgent(env.observation_space, env.action_space, rl_config)

# Training loop with experience collection
def train_rl_agent():
    total_episodes = 1000
    batch_size = 2048
    update_frequency = 10
    
    experience_buffer = []
    
    for episode in range(total_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            # Agent selects action based on current policy
            action, log_prob, value = agent.act(state)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            experience_buffer.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': log_prob,
                'value': value
            })
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done or len(experience_buffer) >= batch_size:
                break
        
        # Update policy when buffer is full
        if len(experience_buffer) >= batch_size or episode % update_frequency == 0:
            # Calculate advantages and returns
            advantages, returns = agent.compute_advantages(experience_buffer)
            
            # Policy update
            policy_loss, value_loss, entropy_loss = agent.update_policy(
                experience_buffer, advantages, returns
            )
            
            # Clear buffer
            experience_buffer = []
            
            # Log training progress
            if episode % 50 == 0:
                print(f"Episode {episode}: Reward={episode_reward:.2f}, "
                      f"PolicyLoss={policy_loss:.4f}, ValueLoss={value_loss:.4f}")
    
    return agent

# Train the agent
trained_agent = train_rl_agent()

# Evaluate performance
evaluation_results = brain.evaluate_agent(trained_agent, env, episodes=100)
print(f"Average reward: {evaluation_results['mean_reward']:.2f}")
print(f"Success rate: {evaluation_results['success_rate']:.2%}")`,

    supervised: `from toolbrain import ToolBrain
from toolbrain.agents import SupervisedAgent
from toolbrain.datasets import load_dataset

# Load and prepare dataset
brain = ToolBrain()
dataset = load_dataset('classification_task')

# Split data
train_data, val_data, test_data = dataset.split(train=0.7, val=0.15, test=0.15)

# Configure supervised learning agent
supervised_config = {
    'model_type': 'neural_network',
    'architecture': {
        'input_size': dataset.feature_dim,
        'hidden_layers': [256, 128, 64],
        'output_size': dataset.num_classes,
        'activation': 'relu',
        'dropout_rate': 0.3,
        'batch_normalization': True
    },
    'optimizer': {
        'type': 'adam',
        'learning_rate': 1e-3,
        'weight_decay': 1e-5
    },
    'loss_function': 'cross_entropy',
    'metrics': ['accuracy', 'f1_score', 'precision', 'recall']
}

# Create supervised agent
agent = SupervisedAgent(supervised_config)

# Training with validation
def train_supervised_agent():
    epochs = 100
    batch_size = 32
    patience = 10
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        # Training phase
        agent.train()
        train_losses = []
        train_accuracies = []
        
        for batch in train_data.get_batches(batch_size):
            # Forward pass
            predictions = agent.predict(batch['features'])
            loss = agent.compute_loss(predictions, batch['labels'])
            
            # Backward pass
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            
            # Track metrics
            accuracy = agent.compute_accuracy(predictions, batch['labels'])
            train_losses.append(loss.item())
            train_accuracies.append(accuracy)
        
        # Validation phase
        agent.eval()
        val_losses = []
        val_accuracies = []
        
        with torch.no_grad():
            for batch in val_data.get_batches(batch_size):
                predictions = agent.predict(batch['features'])
                loss = agent.compute_loss(predictions, batch['labels'])
                accuracy = agent.compute_accuracy(predictions, batch['labels'])
                
                val_losses.append(loss.item())
                val_accuracies.append(accuracy)
        
        # Calculate epoch metrics
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = np.mean(train_accuracies)
        epoch_val_acc = np.mean(val_accuracies)
        
        # Store history
        training_history['train_loss'].append(epoch_train_loss)
        training_history['val_loss'].append(epoch_val_loss)
        training_history['train_acc'].append(epoch_train_acc)
        training_history['val_acc'].append(epoch_val_acc)
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model
            agent.save_checkpoint('best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={epoch_train_loss:.4f}, "
                  f"Val Loss={epoch_val_loss:.4f}, "
                  f"Train Acc={epoch_train_acc:.4f}, "
                  f"Val Acc={epoch_val_acc:.4f}")
    
    return agent, training_history

# Train the agent
trained_agent, history = train_supervised_agent()

# Final evaluation on test set
test_results = brain.evaluate_supervised_agent(trained_agent, test_data)
print(f"Test Accuracy: {test_results['accuracy']:.4f}")
print(f"Test F1-Score: {test_results['f1_score']:.4f}")`,

    unsupervised: `from toolbrain import ToolBrain
from toolbrain.agents import UnsupervisedAgent
from toolbrain.clustering import KMeansClusterer, DBSCANClusterer

# Load unlabeled data
brain = ToolBrain()
data = brain.load_unlabeled_data('customer_behavior.csv')

# Preprocess data
preprocessed_data = brain.preprocess_data(
    data,
    normalize=True,
    handle_missing='impute',
    remove_outliers=True
)

# Configure unsupervised learning
unsupervised_config = {
    'algorithm': 'autoencoder',
    'encoder': {
        'layers': [256, 128, 64, 32],
        'activation': 'relu',
        'dropout_rate': 0.2
    },
    'decoder': {
        'layers': [32, 64, 128, 256],
        'activation': 'relu',
        'output_activation': 'sigmoid'
    },
    'latent_dim': 16,
    'learning_rate': 1e-3,
    'reconstruction_loss': 'mse'
}

# Create unsupervised agent
agent = UnsupervisedAgent(unsupervised_config)

# Training autoencoder for feature learning
def train_autoencoder():
    epochs = 200
    batch_size = 64
    
    reconstruction_losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch in preprocessed_data.get_batches(batch_size):
            # Forward pass through autoencoder
            encoded = agent.encode(batch)
            reconstructed = agent.decode(encoded)
            
            # Compute reconstruction loss
            loss = agent.compute_reconstruction_loss(batch, reconstructed)
            
            # Backward pass
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            
            epoch_losses.append(loss.item())
        
        epoch_loss = np.mean(epoch_losses)
        reconstruction_losses.append(epoch_loss)
        
        if epoch % 25 == 0:
            print(f"Epoch {epoch}: Reconstruction Loss = {epoch_loss:.6f}")
    
    return reconstruction_losses

# Train autoencoder
train_autoencoder()

# Extract learned features
encoded_features = agent.encode(preprocessed_data.features)

# Perform clustering on learned features
clustering_methods = {
    'kmeans': KMeansClusterer(n_clusters=5),
    'dbscan': DBSCANClusterer(eps=0.5, min_samples=10)
}

clustering_results = {}
for method_name, clusterer in clustering_methods.items():
    # Fit clustering algorithm
    clusters = clusterer.fit_predict(encoded_features)
    
    # Evaluate clustering quality
    silhouette_score = clusterer.compute_silhouette_score(encoded_features, clusters)
    davies_bouldin_score = clusterer.compute_davies_bouldin_score(encoded_features, clusters)
    
    clustering_results[method_name] = {
        'clusters': clusters,
        'silhouette_score': silhouette_score,
        'davies_bouldin_score': davies_bouldin_score,
        'n_clusters': len(np.unique(clusters))
    }
    
    print(f"{method_name.upper()} Results:")
    print(f"  Number of clusters: {clustering_results[method_name]['n_clusters']}")
    print(f"  Silhouette score: {silhouette_score:.4f}")
    print(f"  Davies-Bouldin score: {davies_bouldin_score:.4f}")

# Anomaly detection using reconstruction error
reconstruction_errors = agent.compute_reconstruction_errors(preprocessed_data.features)
anomaly_threshold = np.percentile(reconstruction_errors, 95)
anomalies = reconstruction_errors > anomaly_threshold

print(f"\\nAnomaly Detection:")
print(f"Anomaly threshold: {anomaly_threshold:.6f}")
print(f"Number of anomalies detected: {np.sum(anomalies)}")
print(f"Anomaly rate: {np.mean(anomalies):.2%}")`,

    meta: `from toolbrain import ToolBrain
from toolbrain.agents import MetaLearningAgent
from toolbrain.algorithms import MAML

# Configure meta-learning setup
brain = ToolBrain()

# Meta-learning configuration
meta_config = {
    'algorithm': 'MAML',  # Model-Agnostic Meta-Learning
    'base_model': {
        'type': 'neural_network',
        'hidden_layers': [64, 64],
        'activation': 'relu'
    },
    'meta_learning_rate': 1e-3,
    'adaptation_learning_rate': 1e-2,
    'adaptation_steps': 5,
    'meta_batch_size': 16,
    'task_batch_size': 10
}

# Create meta-learning agent
agent = MetaLearningAgent(meta_config)

# Define task distribution for meta-training
class TaskDistribution:
    def __init__(self, task_family='sine_wave'):
        self.task_family = task_family
    
    def sample_task(self):
        if self.task_family == 'sine_wave':
            # Sample sine wave parameters
            amplitude = np.random.uniform(0.1, 5.0)
            phase = np.random.uniform(0, 2*np.pi)
            frequency = np.random.uniform(0.5, 2.0)
            
            def task_function(x):
                return amplitude * np.sin(frequency * x + phase)
            
            return {
                'function': task_function,
                'params': {'amplitude': amplitude, 'phase': phase, 'frequency': frequency}
            }
    
    def generate_task_data(self, task, n_samples=100):
        x = np.random.uniform(-5, 5, n_samples)
        y = task['function'](x)
        return x.reshape(-1, 1), y.reshape(-1, 1)

# Create task distribution
task_dist = TaskDistribution()

# Meta-training loop
def meta_train():
    meta_epochs = 1000
    
    for epoch in range(meta_epochs):
        # Sample batch of tasks
        task_batch = [task_dist.sample_task() for _ in range(meta_config['meta_batch_size'])]
        
        meta_gradients = []
        
        for task in task_batch:
            # Generate support and query sets for this task
            support_x, support_y = task_dist.generate_task_data(task, n_samples=10)
            query_x, query_y = task_dist.generate_task_data(task, n_samples=15)
            
            # Inner loop: adapt to current task
            adapted_params = agent.adapt_to_task(
                support_x, support_y,
                adaptation_steps=meta_config['adaptation_steps'],
                learning_rate=meta_config['adaptation_learning_rate']
            )
            
            # Compute meta-gradient using query set
            query_loss = agent.compute_loss(query_x, query_y, adapted_params)
            meta_grad = agent.compute_meta_gradient(query_loss)
            meta_gradients.append(meta_grad)
        
        # Meta-update: update initial parameters
        average_meta_grad = agent.average_gradients(meta_gradients)
        agent.meta_update(average_meta_grad, meta_config['meta_learning_rate'])
        
        # Logging
        if epoch % 100 == 0:
            # Evaluate on validation tasks
            val_tasks = [task_dist.sample_task() for _ in range(10)]
            val_performance = agent.evaluate_meta_performance(val_tasks)
            
            print(f"Meta-Epoch {epoch}: "
                  f"Val Loss (before adapt): {val_performance['before_adaptation']:.6f}, "
                  f"Val Loss (after adapt): {val_performance['after_adaptation']:.6f}")
    
    return agent

# Perform meta-training
meta_trained_agent = meta_train()

# Test few-shot learning capability
def test_few_shot_learning():
    # Create a completely new task
    test_task = task_dist.sample_task()
    
    # Generate small support set (few-shot)
    support_x, support_y = task_dist.generate_task_data(test_task, n_samples=5)
    test_x, test_y = task_dist.generate_task_data(test_task, n_samples=50)
    
    # Performance before adaptation
    initial_loss = meta_trained_agent.evaluate(test_x, test_y)
    
    # Quick adaptation with few examples
    adapted_agent = meta_trained_agent.adapt_to_task(
        support_x, support_y,
        adaptation_steps=5
    )
    
    # Performance after adaptation
    adapted_loss = adapted_agent.evaluate(test_x, test_y)
    
    print(f"\\nFew-shot Learning Results:")
    print(f"Loss before adaptation: {initial_loss:.6f}")
    print(f"Loss after adaptation (5 examples): {adapted_loss:.6f}")
    print(f"Improvement: {((initial_loss - adapted_loss) / initial_loss * 100):.1f}%")

# Test few-shot learning
test_few_shot_learning()`
  };

  return (
    <Layout>
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-[#E6EDF3] mb-4">Agent Training</h1>
          <p className="text-xl text-gray-400 leading-relaxed">
            Master different training paradigms and create intelligent agents that learn efficiently. 
            Explore reinforcement learning, supervised learning, unsupervised learning, and meta-learning approaches.
          </p>
        </div>

        {/* Training Types Overview */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Training Paradigms</h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {trainingTypes.map((type) => (
              <div
                key={type.id}
                onClick={() => setSelectedTrainingType(type.id)}
                className={`cursor-pointer p-6 rounded-lg border transition-all duration-200 ${
                  selectedTrainingType === type.id
                    ? 'bg-[#58A6FF]/10 border-[#58A6FF]'
                    : 'bg-[#161B22] border-[#30363D] hover:border-[#58A6FF]/50'
                }`}
              >
                <div className="text-3xl mb-3">{type.icon}</div>
                <h3 className="text-lg font-semibold text-[#E6EDF3] mb-2">{type.name}</h3>
                <p className="text-sm text-gray-400 mb-4">{type.description}</p>
                
                <div className="mb-3">
                  <div className="text-xs font-medium text-[#58A6FF] mb-1">ALGORITHMS</div>
                  <div className="flex flex-wrap gap-1">
                    {type.algorithms.slice(0, 3).map((algo, index) => (
                      <span key={index} className="text-xs bg-[#21262D] text-gray-300 px-2 py-1 rounded">
                        {algo}
                      </span>
                    ))}
                    {type.algorithms.length > 3 && (
                      <span className="text-xs text-gray-500">+{type.algorithms.length - 3}</span>
                    )}
                  </div>
                </div>
                
                <div>
                  <div className="text-xs font-medium text-[#3FB950] mb-1">USE CASES</div>
                  <div className="text-xs text-gray-400">
                    {type.useCases.slice(0, 2).join(', ')}
                    {type.useCases.length > 2 && '...'}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Training Type Details */}
        <div className="mb-12">
        {(() => {
          const selectedType = trainingTypes.find(t => t.id === selectedTrainingType);
          if (!selectedType) return null;
          return (
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-8">
              <div className="flex items-center mb-6">
                <div className="text-4xl mr-4">{selectedType.icon}</div>
                <div>
                  <h2 className="text-2xl font-bold text-[#E6EDF3]">{selectedType.name}</h2>
                  <p className="text-gray-400">{selectedType.description}</p>
                </div>
              </div>                <div className="grid md:grid-cols-2 gap-8 mb-8">
                  <div>
                    <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">Available Algorithms</h3>
                    <div className="flex flex-wrap gap-2">
                      {selectedType.algorithms.map((algo, index) => (
                        <span 
                          key={index}
                          className="bg-[#21262D] text-[#E6EDF3] px-3 py-1 rounded-lg text-sm"
                        >
                          {algo}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">Common Use Cases</h3>
                    <ul className="space-y-1">
                      {selectedType.useCases.map((useCase, index) => (
                        <li key={index} className="text-gray-400 text-sm">• {useCase}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">Implementation Example</h3>
                <CodeBlock language="python">
                  {trainingExamples[selectedTrainingType as keyof typeof trainingExamples]}
                </CodeBlock>
              </div>
            );
          })()}
        </div>

        {/* Advanced Training Techniques */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Advanced Training Techniques</h2>
          
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">🚀 Transfer Learning</h3>
                <p className="text-gray-400 mb-4">
                  Leverage pre-trained models and adapt them to new tasks for faster learning.
                </p>
                <CodeBlock language="python">
{`# Load pre-trained model
pretrained_agent = brain.load_pretrained_model('expert_agent.pth')

# Fine-tune for new task
fine_tuned_agent = brain.transfer_learning(
    source_agent=pretrained_agent,
    target_task='new_environment',
    freeze_layers=['encoder'],  # Keep some layers frozen
    learning_rate=1e-4,
    fine_tune_epochs=50
)`}
                </CodeBlock>
              </div>
              
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">⚡ Curriculum Learning</h3>
                <p className="text-gray-400 mb-4">
                  Train agents with progressively difficult tasks for better learning efficiency.
                </p>
                <CodeBlock language="python">
{`# Define curriculum stages
curriculum = brain.create_curriculum([
    {'stage': 'easy', 'episodes': 200, 'difficulty': 0.2},
    {'stage': 'medium', 'episodes': 300, 'difficulty': 0.6},
    {'stage': 'hard', 'episodes': 500, 'difficulty': 1.0}
])

# Train with curriculum
agent = brain.train_with_curriculum(curriculum, agent_config)`}
                </CodeBlock>
              </div>
            </div>
            
            <div className="space-y-6">
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">🔄 Multi-Task Learning</h3>
                <p className="text-gray-400 mb-4">
                  Train a single agent to handle multiple related tasks simultaneously.
                </p>
                <CodeBlock language="python">
{`# Define multiple tasks
tasks = {
    'navigation': NavigationTask(),
    'manipulation': ManipulationTask(),
    'planning': PlanningTask()
}

# Create multi-task agent
multi_agent = brain.create_multi_task_agent(
    tasks=tasks,
    shared_layers=['encoder', 'attention'],
    task_specific_layers=['policy', 'value']
)`}
                </CodeBlock>
              </div>
              
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">🧠 Continual Learning</h3>
                <p className="text-gray-400 mb-4">
                  Enable agents to learn new tasks without forgetting previously learned knowledge.
                </p>
                <CodeBlock language="python">
{`# Configure continual learning
continual_config = {
    'method': 'elastic_weight_consolidation',
    'importance_weight': 0.4,
    'memory_size': 1000,
    'replay_frequency': 10
}

# Train on sequence of tasks
for task in task_sequence:
    agent = brain.continual_learn(agent, task, continual_config)`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </div>

        {/* Training Monitoring */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Training Monitoring & Debugging</h2>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
            <p className="text-gray-400 mb-6">
              Effective monitoring is crucial for successful agent training. ToolBrain provides comprehensive 
              tools to track training progress, identify issues, and optimize performance.
            </p>
            
            <CodeBlock language="python">
{`from toolbrain.monitoring import TrainingMonitor
from toolbrain.visualization import plot_training_curves

# Setup comprehensive monitoring
monitor = TrainingMonitor(
    log_frequency=10,
    save_frequency=100,
    metrics=['reward', 'loss', 'gradient_norm', 'learning_rate'],
    plot_realtime=True
)

# Training with monitoring
def monitored_training_loop():
    for episode in range(1000):
        # Training step
        results = agent.train_step()
        
        # Log metrics
        monitor.log_metrics({
            'episode': episode,
            'reward': results['episode_reward'],
            'loss': results['policy_loss'],
            'value_loss': results['value_loss'],
            'entropy': results['entropy'],
            'gradient_norm': results['grad_norm'],
            'exploration_rate': agent.exploration_rate
        })
        
        # Check for training issues
        if monitor.detect_training_issues():
            issues = monitor.get_detected_issues()
            print(f"Training issues detected: {issues}")
            
            # Auto-adjust learning rate if needed
            if 'gradient_explosion' in issues:
                agent.reduce_learning_rate(factor=0.5)
            elif 'no_improvement' in issues:
                agent.increase_exploration(factor=1.2)
        
        # Save checkpoints
        if episode % 100 == 0:
            monitor.save_checkpoint(agent, f'checkpoint_episode_{episode}.pth')
        
        # Early stopping
        if monitor.should_stop_early():
            print("Early stopping triggered")
            break
    
    # Generate training report
    report = monitor.generate_report()
    print(f"Training completed: {report}")

# Advanced debugging tools    
def debug_agent_behavior():
    # Analyze agent decisions
    decision_analysis = brain.analyze_decisions(agent, test_episodes=50)
    
    # Visualize policy
    brain.visualize_policy(agent, save_path='policy_visualization.png')
    
    # Check gradient flow
    gradient_analysis = brain.analyze_gradients(agent)
    
    # Identify failure modes
    failure_modes = brain.identify_failure_modes(agent, test_cases=100)
    
    return {
        'decisions': decision_analysis,
        'gradients': gradient_analysis,
        'failures': failure_modes
    }

# Run debugging analysis
debug_results = debug_agent_behavior()
print("Debug analysis completed:", debug_results.keys())`}
            </CodeBlock>
          </div>
        </div>

        {/* Best Practices */}
        <div className="bg-gradient-to-r from-[#58A6FF]/10 to-[#3FB950]/10 border border-[#58A6FF]/20 rounded-lg p-8">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">🏆 Training Best Practices</h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">🎯 Training Strategy</h3>
              <ul className="space-y-2 text-gray-300">
                <li>• Start with simple baselines before complex models</li>
                <li>• Use appropriate learning rate schedules</li>
                <li>• Implement proper validation strategies</li>
                <li>• Monitor training stability metrics</li>
                <li>• Save regular checkpoints</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">🔧 Technical Tips</h3>
              <ul className="space-y-2 text-gray-300">
                <li>• Use gradient clipping to prevent explosion</li>
                <li>• Normalize inputs and rewards</li>
                <li>• Implement experience replay for stability</li>
                <li>• Use target networks for Q-learning</li>
                <li>• Apply regularization to prevent overfitting</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}