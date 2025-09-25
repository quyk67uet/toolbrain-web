'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import { useState } from 'react';

export default function HyperparameterOptimization() {
  const [selectedOptimizer, setSelectedOptimizer] = useState('bayesian');

  const optimizerMethods = [
    {
      id: 'bayesian',
      name: 'Bayesian Optimization',
      description: 'Uses Gaussian Processes to model the objective function',
      pros: ['Sample efficient', 'Handles noise well', 'Good for expensive evaluations'],
      cons: ['Slower with high dimensions', 'Complex implementation'],
      bestFor: 'Complex models with expensive training'
    },
    {
      id: 'grid',
      name: 'Grid Search',
      description: 'Exhaustively searches all parameter combinations',
      pros: ['Simple to understand', 'Guaranteed to find best in range', 'Reproducible'],
      cons: ['Exponentially expensive', 'Curse of dimensionality'],
      bestFor: 'Small parameter spaces, initial exploration'
    },
    {
      id: 'random',
      name: 'Random Search',
      description: 'Random sampling from parameter distributions',
      pros: ['More efficient than grid', 'Scales well', 'Easy to parallelize'],
      cons: ['No learning from previous trials', 'May miss optimal regions'],
      bestFor: 'High-dimensional spaces, parallel execution'
    },
    {
      id: 'genetic',
      name: 'Genetic Algorithm',
      description: 'Evolution-inspired population-based optimization',
      pros: ['Global optimization', 'Handles discrete/continuous', 'Population diversity'],
      cons: ['Many hyperparameters', 'Slower convergence'],
      bestFor: 'Multi-modal objectives, mixed parameter types'
    }
  ];

  const codeExamples = {
    bayesian: `from toolbrain import ToolBrain
from toolbrain.optimization import BayesianOptimizer

# Initialize ToolBrain with Bayesian optimization
brain = ToolBrain()

# Define parameter space
param_space = {
    'learning_rate': {'type': 'float', 'range': [1e-5, 1e-1], 'scale': 'log'},
    'batch_size': {'type': 'choice', 'options': [16, 32, 64, 128, 256]},
    'hidden_layers': {'type': 'int', 'range': [1, 5]},
    'hidden_size': {'type': 'int', 'range': [32, 512]},
    'dropout_rate': {'type': 'float', 'range': [0.0, 0.5]},
    'l2_regularization': {'type': 'float', 'range': [1e-6, 1e-2], 'scale': 'log'}
}

# Configure Bayesian optimizer
optimizer_config = {
    'acquisition_function': 'expected_improvement',
    'gaussian_process_kernel': 'matern',
    'n_initial_points': 10,
    'alpha': 1e-6,  # noise parameter
    'normalize_y': True
}

# Create optimizer
optimizer = BayesianOptimizer(param_space, **optimizer_config)

# Define objective function
def train_and_evaluate(params):
    # Configure model with current parameters
    brain.set_config(params)
    
    # Train model
    results = brain.train(
        episodes=100,
        validate_every=20,
        early_stopping_patience=10
    )
    
    # Return negative loss (we want to maximize)
    return -results['validation_loss']

# Run optimization
best_params, best_score, optimization_history = optimizer.optimize(
    objective_function=train_and_evaluate,
    n_trials=50,
    timeout=3600,  # 1 hour timeout
    verbose=True
)

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")`,

    grid: `from toolbrain import ToolBrain
from toolbrain.optimization import GridSearchOptimizer

# Initialize ToolBrain
brain = ToolBrain()

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'hidden_layers': [2, 3, 4],
    'hidden_size': [64, 128, 256],
    'dropout_rate': [0.0, 0.2, 0.4]
}

# Create grid search optimizer
optimizer = GridSearchOptimizer(param_grid)

# Calculate total combinations
total_combinations = optimizer.get_total_combinations()
print(f"Total parameter combinations: {total_combinations}")

# Define evaluation function
def evaluate_params(params):
    brain.set_config(params)
    
    # Quick training for grid search
    results = brain.train(
        episodes=50,
        validate_every=10
    )
    
    return {
        'score': results['final_reward'],
        'training_time': results['training_time'],
        'convergence_episode': results['convergence_episode']
    }

# Run grid search with parallel execution
best_params, best_score, all_results = optimizer.optimize(
    objective_function=evaluate_params,
    n_jobs=4,  # parallel processes
    verbose=True,
    save_results='grid_search_results.json'
)

# Analyze results
print("\\nTop 5 parameter combinations:")
sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
for i, result in enumerate(sorted_results[:5]):
    print(f"{i+1}. Score: {result['score']:.4f}, Params: {result['params']}")`,

    random: `from toolbrain import ToolBrain
from toolbrain.optimization import RandomSearchOptimizer

# Initialize ToolBrain
brain = ToolBrain()

# Define parameter distributions
param_distributions = {
    'learning_rate': {'distribution': 'loguniform', 'low': 1e-5, 'high': 1e-1},
    'batch_size': {'distribution': 'choice', 'options': [16, 32, 64, 128, 256]},
    'hidden_layers': {'distribution': 'randint', 'low': 1, 'high': 6},
    'hidden_size': {'distribution': 'randint', 'low': 32, 'high': 513},
    'dropout_rate': {'distribution': 'uniform', 'low': 0.0, 'high': 0.5},
    'activation': {'distribution': 'choice', 'options': ['relu', 'tanh', 'elu']},
    'optimizer_type': {'distribution': 'choice', 'options': ['adam', 'sgd', 'rmsprop']}
}

# Create random search optimizer
optimizer = RandomSearchOptimizer(param_distributions, random_state=42)

# Define objective with cross-validation
def cross_validate_params(params):
    brain.set_config(params)
    
    # 3-fold cross-validation
    cv_scores = []
    for fold in range(3):
        brain.set_random_seed(fold)
        results = brain.train(
            episodes=100,
            validation_split=0.2
        )
        cv_scores.append(results['validation_score'])
    
    return {
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'scores': cv_scores
    }

# Run random search
results = optimizer.optimize(
    objective_function=cross_validate_params,
    n_trials=100,
    timeout=7200,  # 2 hours
    early_stopping_rounds=20,
    early_stopping_threshold=0.001
)

# Get best parameters
best_params = results['best_params']
best_score = results['best_score']

print(f"Best parameters found:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print(f"\\nBest CV score: {best_score['mean_score']:.4f} ± {best_score['std_score']:.4f}")`
  };

  return (
    <Layout>
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-[#E6EDF3] mb-4">Hyperparameter Optimization</h1>
          <p className="text-xl text-gray-400 leading-relaxed">
            Master the art of hyperparameter optimization with ToolBrain. Learn different strategies, 
            best practices, and advanced techniques to find optimal parameters for your agents.
          </p>
        </div>

        {/* Overview */}
        <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-8 mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Why Hyperparameter Optimization Matters</h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">❌ Without Optimization</h3>
              <ul className="space-y-2 text-gray-400">
                <li>• Manual trial-and-error approach</li>
                <li>• Suboptimal model performance</li>
                <li>• Wasted computational resources</li>
                <li>• Inconsistent results across runs</li>
                <li>• Time-consuming parameter tuning</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">✅ With Optimization</h3>
              <ul className="space-y-2 text-gray-400">
                <li>• Systematic parameter search</li>
                <li>• Maximum model performance</li>
                <li>• Efficient resource utilization</li>
                <li>• Reproducible optimization process</li>
                <li>• Automated parameter discovery</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Optimization Methods */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Optimization Methods</h2>
          
          {/* Method Selector */}
          <div className="flex flex-wrap gap-2 mb-8">
            {optimizerMethods.map((method) => (
              <button
                key={method.id}
                onClick={() => setSelectedOptimizer(method.id)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${
                  selectedOptimizer === method.id
                    ? 'bg-[#58A6FF] text-white'
                    : 'bg-[#161B22] border border-[#30363D] text-gray-400 hover:border-[#58A6FF]'
                }`}
              >
                {method.name}
              </button>
            ))}
          </div>

        {/* Selected Method Details */}
        {(() => {
          const method = optimizerMethods.find(m => m.id === selectedOptimizer);
          if (!method) return null;
          return (
            <div className="grid lg:grid-cols-2 gap-8">
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-3">{method.name}</h3>
                <p className="text-gray-400 mb-6">{method.description}</p>                  <div className="grid md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-semibold text-[#3FB950] mb-2">✅ Pros</h4>
                      <ul className="space-y-1 text-sm text-gray-400">
                        {method.pros.map((pro, index) => (
                          <li key={index}>• {pro}</li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-[#F85149] mb-2">❌ Cons</h4>
                      <ul className="space-y-1 text-sm text-gray-400">
                        {method.cons.map((con, index) => (
                          <li key={index}>• {con}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                  
                  <div className="mt-6 p-3 bg-[#0D1117] rounded-lg">
                    <h4 className="font-semibold text-[#58A6FF] mb-2">🎯 Best For</h4>
                    <p className="text-sm text-gray-300">{method.bestFor}</p>
                  </div>
                </div>

                <div>
                  <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">Implementation Example</h3>
                  <CodeBlock language="python">
                    {codeExamples[selectedOptimizer as keyof typeof codeExamples]}
                  </CodeBlock>
                </div>
              </div>
            );
          })()}
        </div>

        {/* Advanced Techniques */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Advanced Optimization Techniques</h2>
          
          <div className="space-y-8">
            {/* Multi-Objective Optimization */}
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">🎯 Multi-Objective Optimization</h3>
              <p className="text-gray-400 mb-4">
                Optimize for multiple objectives simultaneously (e.g., accuracy vs. speed, performance vs. memory usage).
              </p>
              
              <CodeBlock language="python">
{`from toolbrain.optimization import MultiObjectiveOptimizer

# Define multiple objectives
def multi_objective_function(params):
    brain.set_config(params)
    results = brain.train(episodes=100)
    
    return {
        'accuracy': results['final_accuracy'],      # maximize
        'training_time': -results['training_time'], # minimize (negative)
        'model_size': -results['model_parameters'], # minimize (negative)
        'inference_speed': results['inference_fps'] # maximize
    }

# Create multi-objective optimizer
optimizer = MultiObjectiveOptimizer(
    param_space=param_space,
    objectives=['accuracy', 'training_time', 'model_size', 'inference_speed'],
    weights=[0.4, 0.2, 0.2, 0.2]  # importance weights
)

# Find Pareto optimal solutions
pareto_solutions = optimizer.optimize(
    objective_function=multi_objective_function,
    n_trials=100,
    population_size=20
)`}
              </CodeBlock>
            </div>

            {/* Early Stopping */}
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">⏹️ Smart Early Stopping</h3>
              <p className="text-gray-400 mb-4">
                Stop unpromising parameter combinations early to save computational resources.
              </p>
              
              <CodeBlock language="python">
{`from toolbrain.optimization import EarlyStoppingOptimizer

# Configure early stopping
early_stopping_config = {
    'min_trials': 20,           # minimum trials before stopping
    'patience': 10,             # trials without improvement
    'improvement_threshold': 0.01, # minimum improvement
    'percentile_threshold': 0.3    # stop if below 30th percentile
}

optimizer = EarlyStoppingOptimizer(
    param_space=param_space,
    **early_stopping_config
)

# Training with intermediate reporting
def objective_with_reporting(params):
    brain.set_config(params)
    
    # Report intermediate results
    def intermediate_callback(episode, metrics):
        # Report current performance
        optimizer.report_intermediate_value(episode, metrics['validation_score'])
        
        # Check if trial should be pruned
        if optimizer.should_prune():
            raise optimizer.TrialPruned()
    
    results = brain.train(
        episodes=200,
        intermediate_callback=intermediate_callback
    )
    
    return results['final_score']

# Run optimization with pruning
best_params = optimizer.optimize(objective_with_reporting, n_trials=100)`}
              </CodeBlock>
            </div>

            {/* Distributed Optimization */}
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">🔄 Distributed Optimization</h3>
              <p className="text-gray-400 mb-4">
                Scale optimization across multiple machines or GPUs for faster results.
              </p>
              
              <CodeBlock language="python">
{`from toolbrain.optimization import DistributedOptimizer
from toolbrain.distributed import ClusterManager

# Setup cluster
cluster = ClusterManager()
cluster.add_workers([
    'worker1:8080',
    'worker2:8080', 
    'worker3:8080',
    'worker4:8080'
])

# Create distributed optimizer
optimizer = DistributedOptimizer(
    param_space=param_space,
    cluster=cluster,
    strategy='async',  # or 'sync'
    load_balancing=True
)

# Distribute optimization workload
def distributed_objective(params):
    # This runs on worker nodes
    brain = ToolBrain()
    brain.set_config(params)
    
    # Use local GPU if available
    brain.set_device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = brain.train(episodes=100)
    return results['validation_score']

# Run distributed optimization
best_params = optimizer.optimize(
    objective_function=distributed_objective,
    n_trials=200,
    timeout=3600
)

print(f"Optimization completed using {cluster.get_worker_count()} workers")
print(f"Total computation time saved: {optimizer.get_time_savings():.1f}x")`}
              </CodeBlock>
            </div>
          </div>
        </div>

        {/* Best Practices */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Best Practices & Tips</h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-6">
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">🎯 Parameter Selection</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li>• Focus on parameters with highest impact</li>
                  <li>• Use log scale for learning rates</li>
                  <li>• Start with wider ranges, then narrow down</li>
                  <li>• Consider parameter interactions</li>
                </ul>
              </div>
              
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">⚡ Performance Tips</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li>• Use smaller datasets for initial search</li>
                  <li>• Implement warm-start strategies</li>
                  <li>• Cache expensive computations</li>
                  <li>• Monitor resource utilization</li>
                </ul>
              </div>
            </div>
            
            <div className="space-y-6">
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">📊 Evaluation Strategy</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li>• Use cross-validation for robust estimates</li>
                  <li>• Separate validation from test data</li>
                  <li>• Consider multiple evaluation metrics</li>
                  <li>• Account for random variation</li>
                </ul>
              </div>
              
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">🔍 Analysis & Debugging</h3>
                <ul className="space-y-2 text-gray-400 text-sm">
                  <li>• Visualize parameter importance</li>
                  <li>• Analyze convergence patterns</li>
                  <li>• Check for overfitting to validation</li>
                  <li>• Document optimization process</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Complete Example */}
        <div className="bg-gradient-to-r from-[#58A6FF]/10 to-[#7C3AED]/10 border border-[#58A6FF]/20 rounded-lg p-8">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-4">🚀 Complete Optimization Pipeline</h2>
          <p className="text-gray-300 mb-6">
            Here's a comprehensive example that combines multiple optimization techniques:
          </p>
          
          <CodeBlock language="python">
{`from toolbrain import ToolBrain
from toolbrain.optimization import (
    BayesianOptimizer,
    EarlyStoppingOptimizer, 
    MultiObjectiveOptimizer
)

# Complete optimization pipeline
class OptimizationPipeline:
    def __init__(self, param_space, objectives):
        self.param_space = param_space
        self.objectives = objectives
        self.history = []
    
    def run_optimization(self):
        # Phase 1: Quick exploration with random search
        print("Phase 1: Random exploration...")
        random_results = self.random_search_phase()
        
        # Phase 2: Focused search with Bayesian optimization
        print("Phase 2: Bayesian optimization...")
        bayesian_results = self.bayesian_phase(random_results['top_params'])
        
        # Phase 3: Multi-objective refinement
        print("Phase 3: Multi-objective refinement...")
        final_results = self.multi_objective_phase(bayesian_results['best_params'])
        
        return final_results
    
    def random_search_phase(self):
        # Quick parameter space exploration
        optimizer = RandomSearchOptimizer(self.param_space)
        return optimizer.optimize(self.objective_function, n_trials=50)
    
    def bayesian_phase(self, initial_points):
        # Focused optimization around promising regions
        optimizer = BayesianOptimizer(
            self.param_space,
            initial_points=initial_points
        )
        return optimizer.optimize(self.objective_function, n_trials=100)
    
    def multi_objective_phase(self, best_single_objective):
        # Multi-objective optimization for final tuning
        optimizer = MultiObjectiveOptimizer(
            self.param_space,
            objectives=self.objectives
        )
        return optimizer.optimize(
            self.multi_objective_function, 
            n_trials=50,
            reference_point=best_single_objective
        )
    
    def objective_function(self, params):
        brain = ToolBrain()
        brain.set_config(params)
        results = brain.train(episodes=100)
        return results['validation_score']
    
    def multi_objective_function(self, params):
        brain = ToolBrain()
        brain.set_config(params)
        results = brain.train(episodes=100)
        
        return {
            'accuracy': results['accuracy'],
            'speed': -results['training_time'],
            'efficiency': results['sample_efficiency']
        }

# Run complete optimization
pipeline = OptimizationPipeline(param_space, ['accuracy', 'speed', 'efficiency'])
optimal_results = pipeline.run_optimization()

print("Optimization completed!")
print(f"Best parameters: {optimal_results['best_params']}")
print(f"Pareto front: {len(optimal_results['pareto_solutions'])} solutions")`}
          </CodeBlock>
        </div>
      </div>
    </Layout>
  );
}