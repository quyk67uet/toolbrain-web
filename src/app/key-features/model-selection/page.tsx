'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import { useState } from 'react';

export default function ModelSelection() {
  const [selectedCategory, setSelectedCategory] = useState('neural-networks');

  const modelCategories = [
    {
      id: 'neural-networks',
      name: 'Neural Networks',
      icon: '🧠',
      description: 'Deep learning models for complex pattern recognition',
      models: [
        { name: 'Feedforward Networks', complexity: 'Low', use_case: 'Basic classification' },
        { name: 'Convolutional Networks', complexity: 'Medium', use_case: 'Image processing' },
        { name: 'Recurrent Networks', complexity: 'Medium', use_case: 'Sequential data' },
        { name: 'Transformer Networks', complexity: 'High', use_case: 'Language modeling' },
      ]
    },
    {
      id: 'reinforcement-learning',
      name: 'Reinforcement Learning',
      icon: '🎯',
      description: 'Models that learn through interaction and rewards',
      models: [
        { name: 'Deep Q-Networks (DQN)', complexity: 'Medium', use_case: 'Discrete actions' },
        { name: 'Policy Gradients (PPO)', complexity: 'Medium', use_case: 'Continuous control' },
        { name: 'Actor-Critic (A3C)', complexity: 'High', use_case: 'Parallel training' },
        { name: 'Soft Actor-Critic (SAC)', complexity: 'High', use_case: 'Sample efficiency' },
      ]
    },
    {
      id: 'ensemble-methods',
      name: 'Ensemble Methods',
      icon: '🌟',
      description: 'Combine multiple models for better performance',
      models: [
        { name: 'Random Forest', complexity: 'Low', use_case: 'Tabular data' },
        { name: 'Gradient Boosting', complexity: 'Medium', use_case: 'Structured data' },
        { name: 'Neural Ensembles', complexity: 'High', use_case: 'Uncertainty estimation' },
        { name: 'Multi-Agent Systems', complexity: 'High', use_case: 'Collaborative learning' },
      ]
    },
    {
      id: 'specialized-models',
      name: 'Specialized Models',
      icon: '🔬',
      description: 'Domain-specific architectures for specialized tasks',
      models: [
        { name: 'Graph Neural Networks', complexity: 'High', use_case: 'Graph data' },
        { name: 'Attention Mechanisms', complexity: 'Medium', use_case: 'Focus learning' },
        { name: 'Memory Networks', complexity: 'High', use_case: 'Long-term memory' },
        { name: 'Meta-Learning Models', complexity: 'High', use_case: 'Few-shot learning' },
      ]
    }
  ];

  const selectionCriteria = [
    {
      factor: 'Data Type',
      description: 'Nature of your input data',
      considerations: [
        'Tabular: Tree-based models, neural networks',
        'Image: CNNs, Vision Transformers',
        'Text: RNNs, Transformers',
        'Sequential: RNNs, LSTMs, GRUs',
        'Graph: Graph Neural Networks'
      ]
    },
    {
      factor: 'Problem Type',
      description: 'Category of machine learning task',
      considerations: [
        'Classification: Discriminative models',
        'Regression: Continuous output models',
        'Generation: Generative models',
        'Control: Reinforcement learning',
        'Clustering: Unsupervised models'
      ]
    },
    {
      factor: 'Dataset Size',
      description: 'Amount of available training data',
      considerations: [
        'Small (< 1K): Simple models, avoid overfitting',
        'Medium (1K-100K): Traditional ML, small NNs',
        'Large (100K-1M): Deep networks, ensembles',
        'Very Large (> 1M): Large-scale deep learning',
        'Few-shot: Meta-learning, transfer learning'
      ]
    },
    {
      factor: 'Performance Requirements',
      description: 'Speed and accuracy constraints',
      considerations: [
        'Real-time inference: Lightweight models',
        'High accuracy: Complex ensembles',
        'Low latency: Optimized architectures',
        'Resource constraints: Mobile-friendly models',
        'Scalability: Distributed architectures'
      ]
    }
  ];

  return (
    <Layout>
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-[#E6EDF3] mb-4">Model Selection & Architecture</h1>
          <p className="text-xl text-gray-400 leading-relaxed">
            Choose the right model architecture for your specific task. ToolBrain provides intelligent 
            recommendations and automated model selection to optimize performance for your use case.
          </p>
        </div>

        {/* Model Factory Overview */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">ToolBrain Model Factory</h2>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-8 mb-8">
            <p className="text-gray-400 mb-6">
              The ModelFactory provides a unified interface for creating, configuring, and optimizing 
              different model architectures. It automatically selects appropriate models based on your 
              data characteristics and task requirements.
            </p>
            
            <CodeBlock language="python">
{`from toolbrain import ModelFactory, ToolBrain

# Initialize model factory
factory = ModelFactory()
brain = ToolBrain()

# Automatic model selection based on data
model = factory.auto_select(
    data_type='tabular',
    task_type='classification',
    dataset_size=50000,
    performance_priority='accuracy',  # or 'speed', 'memory'
    constraints={'max_inference_time': 100}  # milliseconds
)

print(f"Selected model: {model.architecture}")
print(f"Expected performance: {model.estimated_metrics}")

# Manual model creation with custom configuration
custom_model = factory.create_model(
    model_type='transformer',
    config={
        'hidden_size': 768,
        'num_layers': 12,
        'attention_heads': 12,
        'dropout_rate': 0.1,
        'activation': 'gelu'
    }
)

# Model comparison and selection
candidates = factory.compare_models(
    models=['random_forest', 'xgboost', 'neural_network'],
    evaluation_metric='f1_score',
    cross_validation_folds=5
)

best_model = candidates.get_best_model()
print(f"Best model: {best_model.name} (F1: {best_model.f1_score:.4f})")`}
            </CodeBlock>
          </div>
        </div>

        {/* Model Categories */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Model Categories</h2>
          
          {/* Category Selector */}
          <div className="flex flex-wrap gap-2 mb-8">
            {modelCategories.map((category) => (
              <button
                key={category.id}
                onClick={() => setSelectedCategory(category.id)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 flex items-center gap-2 ${
                  selectedCategory === category.id
                    ? 'bg-[#58A6FF] text-white'
                    : 'bg-[#161B22] border border-[#30363D] text-gray-400 hover:border-[#58A6FF]'
                }`}
              >
                <span>{category.icon}</span>
                {category.name}
              </button>
            ))}
          </div>

          {/* Selected Category Details */}
          {(() => {
            const category = modelCategories.find(c => c.id === selectedCategory);
            if (!category) return null;
            return (
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-8">
                <div className="flex items-center mb-6">
                  <div className="text-3xl mr-4">{category.icon}</div>
                  <div>
                    <h3 className="text-2xl font-bold text-[#E6EDF3]">{category.name}</h3>
                    <p className="text-gray-400">{category.description}</p>
                  </div>
                </div>

                <div className="grid md:grid-cols-2 gap-6">
                  {category.models.map((model, index) => (
                    <div key={index} className="bg-[#0D1117] rounded-lg p-6">
                      <h4 className="text-lg font-semibold text-[#E6EDF3] mb-2">{model.name}</h4>
                      <div className="flex items-center justify-between mb-3">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          model.complexity === 'Low' 
                            ? 'bg-[#3FB950] text-white'
                            : model.complexity === 'Medium'
                            ? 'bg-[#FB8500] text-white'
                            : 'bg-[#F85149] text-white'
                        }`}>
                          {model.complexity} Complexity
                        </span>
                      </div>
                      <p className="text-gray-400 text-sm">{model.use_case}</p>
                    </div>
                  ))}
                </div>
              </div>
            );
          })()}
        </div>

        {/* Selection Criteria */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Model Selection Criteria</h2>
          
          <div className="grid lg:grid-cols-2 gap-8">
            {selectionCriteria.map((criteria, index) => (
              <div key={index} className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-3">{criteria.factor}</h3>
                <p className="text-gray-400 mb-4">{criteria.description}</p>
                
                <ul className="space-y-2">
                  {criteria.considerations.map((consideration, cIndex) => (
                    <li key={cIndex} className="text-sm text-gray-300 flex items-start">
                      <span className="text-[#58A6FF] mr-2">•</span>
                      {consideration}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        {/* Automated Selection Process */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Automated Model Selection</h2>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-8">
            <p className="text-gray-400 mb-6">
              ToolBrain implements sophisticated model selection algorithms that automatically 
              choose and configure the best model architecture for your specific task.
            </p>
            
            <CodeBlock language="python">
{`from toolbrain import ModelSelector, DataAnalyzer

# Analyze your dataset
analyzer = DataAnalyzer()
data_profile = analyzer.analyze_dataset(
    data=your_dataset,
    target_column='label',
    analyze_distribution=True,
    detect_patterns=True
)

print(f"Dataset characteristics:")
print(f"  - Size: {data_profile.size}")
print(f"  - Features: {data_profile.num_features}")
print(f"  - Data types: {data_profile.feature_types}")
print(f"  - Missing values: {data_profile.missing_percentage}%")
print(f"  - Class balance: {data_profile.class_distribution}")

# Automated model selection
selector = ModelSelector()

# Define search space and constraints
search_config = {
    'model_families': ['neural_networks', 'tree_based', 'ensemble'],
    'max_training_time': 3600,  # 1 hour
    'max_model_size': '100MB',
    'target_metric': 'f1_score',
    'cv_folds': 5
}

# Run automated selection
results = selector.find_best_model(
    dataset=your_dataset,
    task_type='classification',
    search_config=search_config,
    data_profile=data_profile
)

print(f"\\nBest model found:")
print(f"  - Architecture: {results.best_model.architecture}")
print(f"  - Performance: {results.best_model.cv_score:.4f}")
print(f"  - Training time: {results.best_model.training_time:.1f}s")
print(f"  - Model size: {results.best_model.size}")

# Get top 3 candidates with explanations
top_candidates = results.get_top_k(k=3)
for i, candidate in enumerate(top_candidates):
    print(f"\\nCandidate {i+1}: {candidate.name}")
    print(f"  Score: {candidate.score:.4f}")
    print(f"  Reasoning: {candidate.selection_reasoning}")
    print(f"  Trade-offs: {candidate.trade_offs}")

# Detailed model analysis
best_model = results.best_model
analysis = selector.analyze_model_choice(best_model, data_profile)

print(f"\\nModel Analysis:")
print(f"  - Strengths: {analysis.strengths}")
print(f"  - Weaknesses: {analysis.weaknesses}")
print(f"  - Recommendations: {analysis.recommendations}")
print(f"  - Expected performance on new data: {analysis.generalization_estimate}")`}
            </CodeBlock>
          </div>
        </div>

        {/* Advanced Features */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Advanced Model Features</h2>
          
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">🔄 Neural Architecture Search</h3>
                <p className="text-gray-400 mb-4">
                  Automatically discover optimal neural network architectures for your specific task.
                </p>
                <CodeBlock language="python">
{`# Neural Architecture Search
nas = NeuralArchitectureSearch()

# Define search space
search_space = {
    'layers': [2, 3, 4, 5],
    'layer_sizes': [64, 128, 256, 512],
    'activations': ['relu', 'gelu', 'swish'],
    'dropout_rates': [0.0, 0.1, 0.2, 0.3],
    'batch_norm': [True, False]
}

# Search for optimal architecture
best_architecture = nas.search(
    search_space=search_space,
    dataset=train_data,
    validation_data=val_data,
    search_strategy='evolutionary',
    max_trials=100,
    objective='val_accuracy'
)

print(f"Optimal architecture: {best_architecture}")`}
                </CodeBlock>
              </div>

              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">📊 Model Interpretability</h3>
                <p className="text-gray-400 mb-4">
                  Understand how your models make decisions with built-in interpretability tools.
                </p>
                <CodeBlock language="python">
{`# Model interpretability
interpreter = ModelInterpreter(model)

# Feature importance
importance = interpreter.feature_importance(
    method='permutation',  # or 'shap', 'lime'
    test_data=test_set
)

# Generate explanations
explanations = interpreter.explain_predictions(
    instances=sample_data,
    explanation_type='local',  # or 'global'
    visualization=True
)

# Model behavior analysis
behavior = interpreter.analyze_behavior(
    decision_boundaries=True,
    activation_patterns=True,
    attention_weights=True
)`}
                </CodeBlock>
              </div>
            </div>
            
            <div className="space-y-6">
              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">⚡ Model Optimization</h3>
                <p className="text-gray-400 mb-4">
                  Optimize models for deployment with quantization, pruning, and distillation.
                </p>
                <CodeBlock language="python">
{`# Model optimization for deployment
optimizer = ModelOptimizer()

# Quantization for faster inference
quantized_model = optimizer.quantize(
    model=trained_model,
    quantization_type='int8',  # or 'fp16'
    calibration_data=calib_set
)

# Pruning for smaller models
pruned_model = optimizer.prune(
    model=trained_model,
    pruning_ratio=0.3,
    pruning_strategy='magnitude'  # or 'structured'
)

# Knowledge distillation
student_model = optimizer.distill(
    teacher_model=large_model,
    student_architecture='small_net',
    temperature=3.0,
    alpha=0.7
)

print(f"Original size: {trained_model.size}")
print(f"Optimized size: {quantized_model.size}")
print(f"Speed improvement: {optimizer.benchmark_speed()}")`}
                </CodeBlock>
              </div>

              <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-4">🎯 Multi-Objective Optimization</h3>
                <p className="text-gray-400 mb-4">
                  Balance multiple objectives like accuracy, speed, and memory usage.
                </p>
                <CodeBlock language="python">
{`# Multi-objective model selection
multi_optimizer = MultiObjectiveSelector()

# Define objectives and weights
objectives = {
    'accuracy': {'weight': 0.5, 'direction': 'maximize'},
    'inference_speed': {'weight': 0.3, 'direction': 'maximize'},
    'model_size': {'weight': 0.2, 'direction': 'minimize'}
}

# Find Pareto optimal models
pareto_models = multi_optimizer.optimize(
    model_candidates=candidate_models,
    objectives=objectives,
    test_data=test_set
)

# Select final model based on deployment constraints
final_model = multi_optimizer.select_for_deployment(
    pareto_models=pareto_models,
    constraints={
        'max_latency': 50,  # ms
        'max_memory': 512,  # MB
        'min_accuracy': 0.85
    }
)

print(f"Selected model trade-offs:")
print(f"  Accuracy: {final_model.accuracy:.3f}")
print(f"  Speed: {final_model.inference_speed:.1f} req/s")
print(f"  Size: {final_model.size} MB")`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </div>

        {/* Best Practices */}
        <div className="bg-gradient-to-r from-[#58A6FF]/10 to-[#3FB950]/10 border border-[#58A6FF]/20 rounded-lg p-8">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">🏆 Model Selection Best Practices</h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">✅ Do's</h3>
              <ul className="space-y-2 text-gray-300">
                <li>• Start with simple baselines before complex models</li>
                <li>• Consider data size and computational constraints</li>
                <li>• Use cross-validation for robust model comparison</li>
                <li>• Profile models for production requirements</li>
                <li>• Document model selection rationale</li>
                <li>• Test multiple architectures systematically</li>
                <li>• Consider model interpretability requirements</li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-lg font-semibold text-[#E6EDF3] mb-4">❌ Don'ts</h3>
              <ul className="space-y-2 text-gray-300">
                <li>• Don't choose models based on hype alone</li>
                <li>• Don't ignore computational costs</li>
                <li>• Don't overfit to validation metrics</li>
                <li>• Don't neglect model maintenance needs</li>
                <li>• Don't skip baseline comparisons</li>
                <li>• Don't ignore data preprocessing requirements</li>
                <li>• Don't forget about model explainability</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}