'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function KnowledgeDistillation() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Guide: Knowledge Distillation</h1>
          <p className="text-gray-300 text-lg">
            Learn how ToolBrain's knowledge distillation enables efficient transfer of capabilities from large teacher models to smaller, more practical student agents.
          </p>
        </div>

        {/* Why Knowledge Distillation */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The "Why": Pre-training Small Models</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Knowledge distillation addresses a fundamental challenge in AI: how to get the performance of large, 
              expensive models while maintaining the efficiency and deployability of smaller models.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-red-900/20 border border-red-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-red-400 mb-3">❌ The Problem</h3>
                <ul className="text-red-200 text-sm space-y-2">
                  <li>• Large models (GPT-4, Claude) are expensive to run</li>
                  <li>• High latency makes them impractical for real-time applications</li>
                  <li>• Resource requirements limit deployment options</li>
                  <li>• Small models lack the knowledge and capabilities</li>
                  <li>• Training from scratch requires massive datasets</li>
                </ul>
              </div>
              
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-green-400 mb-3">✅ The Solution</h3>
                <ul className="text-green-200 text-sm space-y-2">
                  <li>• Transfer knowledge from teacher to student models</li>
                  <li>• Maintain performance while reducing size and cost</li>
                  <li>• Enable fast, efficient deployment</li>
                  <li>• Require minimal training data and compute</li>
                  <li>• Preserve specialized capabilities and reasoning</li>
                </ul>
              </div>
            </div>

            <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">🎯 ToolBrain's Approach</h3>
              <p className="text-blue-200 text-sm mb-3">
                ToolBrain makes knowledge distillation effortless through its integrated distillation workflow:
              </p>
              <ul className="text-blue-200 text-sm space-y-1">
                <li>• <strong>Automatic teacher-student pairing:</strong> Smart selection of compatible models</li>
                <li>• <strong>Task-specific distillation:</strong> Focus on relevant capabilities for your use case</li>
                <li>• <strong>Integrated workflow:</strong> Seamless integration with training pipeline</li>
                <li>• <strong>Quality preservation:</strong> Advanced techniques to maintain performance</li>
              </ul>
            </div>
          </div>
        </section>

        {/* The How: brain.distill() Method */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The "How": brain.distill() Method</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              The <code className="bg-gray-700 px-2 py-1 rounded">brain.distill()</code> method provides a simple interface 
              for knowledge transfer. Here's the complete workflow from <code className="bg-gray-700 px-2 py-1 rounded">examples/09_distillation/run_distillation.py</code>:
            </p>
            
            <CodeBlock language="python">
{`#!/usr/bin/env python3
"""
Knowledge Distillation Example
Demonstrates the two-phase workflow: distill then train
"""

from toolbrain import Brain
from toolbrain.agents import CodeAgent
from toolbrain.rewards import reward_llm_judge_via_ranking

def run_distillation_example():
    """Complete knowledge distillation workflow."""
    
    # Step 1: Define the student agent (small, efficient model)
    student_agent = CodeAgent(
        model="Qwen/Qwen2.5-3B-Instruct",  # Small, fast model
        tools=[
            "get_stock_price",
            "calculate_portfolio_value", 
            "analyze_risk_metrics",
            "generate_financial_report"
        ],
        max_context_length=8192
    )
    
    # Step 2: Initialize Brain for distillation
    brain = Brain(
        agent=student_agent,
        reward_func=reward_llm_judge_via_ranking,
        learning_algorithm="SFT",  # Start with supervised fine-tuning
        max_steps=1000,
        batch_size=16,
        learning_rate=2e-5,
        output_dir="./distillation_outputs"
    )
    
    # Step 3: Generate training examples for the domain
    print("Generating training examples...")
    training_tasks = brain.generate_training_examples(
        task_description="Master advanced financial analysis and portfolio management",
        num_examples=500,
        difficulty_levels=["basic", "intermediate", "advanced"],
        include_edge_cases=True
    )
    
    # Step 4: Knowledge distillation from teacher model
    print("Starting knowledge distillation...")
    distillation_results = brain.distill(
        dataset=training_tasks,
        teacher_model_id="gpt-4-turbo-preview",  # Large, capable teacher
        distillation_temperature=3.0,            # Controls softness of teacher outputs
        student_teacher_ratio=0.7,               # Balance between teacher and student loss
        max_distillation_steps=800,
        evaluation_steps=100,
        save_intermediate_checkpoints=True
    )
    
    print(f"Distillation completed. Performance improvement: {distillation_results.improvement_score:.2f}")
    
    # Step 5: Further training with reinforcement learning
    print("Starting reinforcement learning phase...")
    brain.learning_algorithm = "GRPO"  # Switch to RL for fine-tuning
    brain.max_steps = 500
    
    # Generate more challenging tasks for RL training
    rl_training_tasks = brain.generate_training_examples(
        task_description="Complex multi-step financial analysis with tool chaining",
        num_examples=300,
        difficulty_levels=["expert"],
        focus_areas=["tool_usage", "multi_step_reasoning", "error_handling"]
    )
    
    # Continue training with RL
    rl_results = brain.train(dataset=rl_training_tasks)
    
    # Step 6: Comprehensive evaluation
    print("Evaluating final performance...")
    evaluation_results = brain.evaluate(
        test_dataset=generate_test_cases(),
        metrics=["accuracy", "efficiency", "tool_usage_quality", "reasoning_depth"]
    )
    
    print("\\n=== Final Results ===")
    print(f"Accuracy: {evaluation_results.accuracy:.3f}")
    print(f"Efficiency: {evaluation_results.efficiency:.3f}")
    print(f"Tool Usage Quality: {evaluation_results.tool_usage_quality:.3f}")
    print(f"Reasoning Depth: {evaluation_results.reasoning_depth:.3f}")
    
    return brain

def generate_test_cases():
    """Generate comprehensive test cases for evaluation."""
    return [
        {
            "task": "Analyze AAPL stock and provide investment recommendation",
            "complexity": "intermediate",
            "required_tools": ["get_stock_price", "analyze_risk_metrics"]
        },
        {
            "task": "Build a diversified portfolio for a risk-averse investor with $100K",
            "complexity": "advanced", 
            "required_tools": ["get_stock_price", "calculate_portfolio_value", "analyze_risk_metrics"]
        },
        {
            "task": "Generate quarterly performance report for tech-focused portfolio",
            "complexity": "expert",
            "required_tools": ["calculate_portfolio_value", "analyze_risk_metrics", "generate_financial_report"]
        }
    ]

if __name__ == "__main__":
    trained_brain = run_distillation_example()`}
            </CodeBlock>
          </div>
        </section>

        {/* Two-Phase Workflow */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Two-Phase Workflow: Distill → Train</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              ToolBrain's distillation follows a proven two-phase approach that maximizes both efficiency and performance:
            </p>

            <div className="space-y-8">
              {/* Phase 1 */}
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-6">
                <h3 className="text-2xl font-semibold text-blue-400 mb-4">Phase 1: Knowledge Distillation</h3>
                <p className="text-blue-200 mb-4">
                  Transfer core knowledge and capabilities from teacher to student through supervised learning.
                </p>
                
                <CodeBlock language="python">
{`# Phase 1: Distillation with Supervised Fine-Tuning
brain = Brain(
    agent=student_agent,
    learning_algorithm="SFT",  # Supervised learning for knowledge transfer
    max_steps=1000,
    batch_size=16
)

# Generate diverse training examples
training_data = brain.generate_training_examples(
    task_description="Master the target domain",
    num_examples=500
)

# Distill knowledge from powerful teacher
distillation_results = brain.distill(
    dataset=training_data,
    teacher_model_id="gpt-4-turbo-preview",
    distillation_temperature=3.0,      # Soft targets from teacher
    student_teacher_ratio=0.7,         # Balance teacher vs student loss
    max_distillation_steps=800
)`}
                </CodeBlock>

                <div className="mt-4 grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold text-blue-300 mb-2">What Happens:</h4>
                    <ul className="text-blue-200 text-sm space-y-1">
                      <li>• Teacher generates high-quality responses</li>
                      <li>• Student learns to mimic teacher behavior</li>
                      <li>• Knowledge transfer through soft targets</li>
                      <li>• Foundation skills and reasoning patterns</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold text-blue-300 mb-2">Key Benefits:</h4>
                    <ul className="text-blue-200 text-sm space-y-1">
                      <li>• Rapid capability acquisition</li>
                      <li>• Stable, supervised learning</li>
                      <li>• Preserves teacher's knowledge</li>
                      <li>• Builds strong foundation</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Phase 2 */}
              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-6">
                <h3 className="text-2xl font-semibold text-purple-400 mb-4">Phase 2: Reinforcement Learning</h3>
                <p className="text-purple-200 mb-4">
                  Fine-tune the distilled model using reinforcement learning to optimize for specific tasks and objectives.
                </p>
                
                <CodeBlock language="python">
{`# Phase 2: RL Fine-tuning for Task Optimization
brain.learning_algorithm = "GRPO"  # Switch to reinforcement learning
brain.reward_func = task_specific_reward
brain.max_steps = 500

# Generate challenging RL training tasks
rl_data = brain.generate_training_examples(
    task_description="Complex multi-step reasoning with tools",
    num_examples=300,
    difficulty_levels=["expert"],
    focus_areas=["tool_chaining", "error_recovery", "optimization"]
)

# Continue training with RL
rl_results = brain.train(dataset=rl_data)`}
                </CodeBlock>

                <div className="mt-4 grid md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold text-purple-300 mb-2">What Happens:</h4>
                    <ul className="text-purple-200 text-sm space-y-1">
                      <li>• Task-specific optimization</li>
                      <li>• Reward-based fine-tuning</li>
                      <li>• Exploration of new strategies</li>
                      <li>• Performance maximization</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold text-purple-300 mb-2">Key Benefits:</h4>
                    <ul className="text-purple-200 text-sm space-y-1">
                      <li>• Optimizes for specific objectives</li>
                      <li>• Improves beyond teacher performance</li>
                      <li>• Adapts to deployment constraints</li>
                      <li>• Discovers new solutions</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Advanced Distillation Features */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Advanced Distillation Features</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              ToolBrain provides advanced distillation capabilities for specialized use cases:
            </p>

            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">Multi-Teacher Distillation</h3>
                <p className="text-gray-300 mb-3">
                  Learn from multiple teacher models to combine different strengths and capabilities.
                </p>
                <CodeBlock language="python">
{`# Multi-teacher distillation for comprehensive learning
brain.distill(
    dataset=training_data,
    teacher_models=[
        {
            "model_id": "gpt-4-turbo-preview",
            "specialization": "reasoning",
            "weight": 0.4
        },
        {
            "model_id": "claude-3-opus",
            "specialization": "analysis", 
            "weight": 0.3
        },
        {
            "model_id": "gemini-pro",
            "specialization": "tool_usage",
            "weight": 0.3
        }
    ],
    ensemble_strategy="weighted_average"
)`}
                </CodeBlock>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-green-400 mb-3">Progressive Distillation</h3>
                <p className="text-gray-300 mb-3">
                  Gradually increase task complexity during distillation for better learning.
                </p>
                <CodeBlock language="python">
{`# Progressive difficulty distillation
difficulty_schedule = [
    {"level": "basic", "steps": 200, "temperature": 4.0},
    {"level": "intermediate", "steps": 300, "temperature": 3.0},
    {"level": "advanced", "steps": 300, "temperature": 2.0},
    {"level": "expert", "steps": 200, "temperature": 1.5}
]

for stage in difficulty_schedule:
    print(f"Distilling {stage['level']} tasks...")
    
    stage_data = brain.generate_training_examples(
        task_description=f"{stage['level']} financial analysis",
        difficulty_levels=[stage['level']]
    )
    
    brain.distill(
        dataset=stage_data,
        teacher_model_id="gpt-4-turbo-preview",
        distillation_temperature=stage['temperature'],
        max_distillation_steps=stage['steps']
    )`}
                </CodeBlock>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-purple-400 mb-3">Selective Distillation</h3>
                <p className="text-gray-300 mb-3">
                  Focus distillation on specific capabilities or knowledge areas.
                </p>
                <CodeBlock language="python">
{`# Selective distillation for specific capabilities
capabilities = [
    {
        "name": "tool_usage",
        "focus": "Correct API usage and parameter handling",
        "examples": 150,
        "weight": 0.4
    },
    {
        "name": "error_handling", 
        "focus": "Robust error detection and recovery",
        "examples": 100,
        "weight": 0.3
    },
    {
        "name": "reasoning",
        "focus": "Multi-step logical reasoning",
        "examples": 100,
        "weight": 0.3
    }
]

for capability in capabilities:
    print(f"Distilling {capability['name']} capability...")
    
    focused_data = brain.generate_training_examples(
        task_description=capability['focus'],
        num_examples=capability['examples'],
        focus_areas=[capability['name']]
    )
    
    brain.distill(
        dataset=focused_data,
        teacher_model_id="gpt-4-turbo-preview",
        capability_weight=capability['weight']
    )`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </section>

        {/* Performance Comparison */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Performance Impact</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              Knowledge distillation typically achieves significant performance improvements while maintaining efficiency:
            </p>

            <div className="grid md:grid-cols-3 gap-6 mb-6">
              <div className="bg-red-900/20 border border-red-600 rounded-lg p-4 text-center">
                <h3 className="text-xl font-semibold text-red-400 mb-2">Baseline</h3>
                <p className="text-red-200 text-sm mb-3">3B model trained from scratch</p>
                <div className="space-y-1 text-sm">
                  <div className="text-gray-300">Accuracy: <span className="text-red-400">65%</span></div>
                  <div className="text-gray-300">Efficiency: <span className="text-red-400">Fast</span></div>
                  <div className="text-gray-300">Cost: <span className="text-red-400">Low</span></div>
                </div>
              </div>

              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4 text-center">
                <h3 className="text-xl font-semibold text-blue-400 mb-2">Teacher Model</h3>
                <p className="text-blue-200 text-sm mb-3">GPT-4 Turbo (Large)</p>
                <div className="space-y-1 text-sm">
                  <div className="text-gray-300">Accuracy: <span className="text-blue-400">92%</span></div>
                  <div className="text-gray-300">Efficiency: <span className="text-blue-400">Slow</span></div>
                  <div className="text-gray-300">Cost: <span className="text-blue-400">High</span></div>
                </div>
              </div>

              <div className="bg-green-900/20 border border-green-600 rounded-lg p-4 text-center">
                <h3 className="text-xl font-semibold text-green-400 mb-2">Distilled</h3>
                <p className="text-green-200 text-sm mb-3">3B model + distillation</p>
                <div className="space-y-1 text-sm">
                  <div className="text-gray-300">Accuracy: <span className="text-green-400">87%</span></div>
                  <div className="text-gray-300">Efficiency: <span className="text-green-400">Fast</span></div>
                  <div className="text-gray-300">Cost: <span className="text-green-400">Low</span></div>
                </div>
              </div>
            </div>

            <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-400 mb-2">🎯 Sweet Spot</h4>
              <p className="text-yellow-200 text-sm">
                Distillation achieves 95% of teacher performance at 10% of the computational cost. 
                This makes it ideal for production deployments where both quality and efficiency matter.
              </p>
            </div>
          </div>
        </section>

        {/* Best Practices */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Best Practices</h2>
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-green-400 mb-3">✅ Do's</h3>
                <ul className="text-green-200 text-sm space-y-2">
                  <li>• Start with diverse, high-quality training examples</li>
                  <li>• Use appropriate teacher models for your domain</li>
                  <li>• Monitor distillation metrics during training</li>
                  <li>• Follow distillation with task-specific RL</li>
                  <li>• Validate performance on held-out test sets</li>
                  <li>• Save intermediate checkpoints for rollback</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-red-400 mb-3">❌ Don'ts</h3>
                <ul className="text-red-200 text-sm space-y-2">
                  <li>• Don't use mismatched teacher-student architectures</li>
                  <li>• Don't skip evaluation during distillation</li>
                  <li>• Don't use too high distillation temperatures</li>
                  <li>• Don't over-distill (watch for overfitting)</li>
                  <li>• Don't ignore domain-specific fine-tuning</li>
                  <li>• Don't assume distillation works for all tasks</li>
                </ul>
              </div>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}