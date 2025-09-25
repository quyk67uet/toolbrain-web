'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function MultipleAlgorithms() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Guide: Multiple Learning Algorithms</h1>
          <p className="text-gray-300 text-lg">
            Explore ToolBrain's support for different learning algorithms, from reinforcement learning to supervised fine-tuning, with easy algorithm switching.
          </p>
        </div>

        {/* Algorithm Overview */}
        <section className="mb-12">
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              ToolBrain supports multiple learning algorithms, each optimized for different types of tasks and training scenarios. 
              You can easily switch between algorithms using the <code className="bg-gray-700 px-2 py-1 rounded">learning_algorithm</code> 
              parameter in the Brain constructor.
            </p>
            
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-blue-400 mb-2">🎯 GRPO</h3>
                <p className="text-blue-200 text-sm">
                  Generalized Reinforcement Policy Optimization. Best for complex reasoning tasks and tool usage scenarios.
                </p>
              </div>
              
              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-purple-400 mb-2">🔄 DPO</h3>
                <p className="text-purple-200 text-sm">
                  Direct Preference Optimization. Ideal for tasks with comparative feedback and ranking-based rewards.
                </p>
              </div>
              
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-green-400 mb-2">📚 SFT</h3>
                <p className="text-green-200 text-sm">
                  Supervised Fine-Tuning. Perfect for knowledge distillation and learning from expert demonstrations.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Easy Algorithm Switching */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Easy Algorithm Switching</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Switching between algorithms is as simple as changing the <code className="bg-gray-700 px-2 py-1 rounded">learning_algorithm</code> 
              parameter in the Brain's <code className="bg-gray-700 px-2 py-1 rounded">__init__</code> method:
            </p>
            
            <CodeBlock language="python">
{`# From Brain.__init__ method in toolbrain/brain.py
class Brain:
    def __init__(
        self,
        agent,
        reward_func=None,
        learning_algorithm="GRPO",  # <- Easy algorithm selection
        max_steps=1000,
        batch_size=32,
        learning_rate=1e-4,
        **kwargs
    ):
        """
        Initialize Brain with specified learning algorithm.
        
        Args:
            learning_algorithm (str): Algorithm to use for training
                - "GRPO": Generalized Reinforcement Policy Optimization
                - "DPO": Direct Preference Optimization  
                - "SFT": Supervised Fine-Tuning
        """
        self.learning_algorithm = learning_algorithm
        self._setup_trainer(learning_algorithm)
    
    def _setup_trainer(self, algorithm):
        """Configure trainer based on selected algorithm."""
        if algorithm == "GRPO":
            self.trainer = GRPOTrainer(
                model=self.agent.model,
                reward_func=self.reward_func,
                **self.training_config
            )
        elif algorithm == "DPO":
            self.trainer = DPOTrainer(
                model=self.agent.model,
                preference_data=self.preference_data,
                **self.training_config
            )
        elif algorithm == "SFT":
            self.trainer = SFTTrainer(
                model=self.agent.model,
                dataset=self.supervised_dataset,
                **self.training_config
            )`}
            </CodeBlock>

            <div className="mt-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Algorithm Switching Examples</h3>
              <CodeBlock language="python">
{`# Same agent, different algorithms
agent = CodeAgent(model="Qwen/3B-Instruct", tools=finance_tools)

# GRPO for complex reasoning
brain_grpo = Brain(
    agent=agent,
    reward_func=complex_reasoning_reward,
    learning_algorithm="GRPO",
    max_steps=1000
)

# DPO for preference learning
brain_dpo = Brain(
    agent=agent,
    reward_func=reward_llm_judge_via_ranking,
    learning_algorithm="DPO",
    max_steps=800
)

# SFT for knowledge transfer
brain_sft = Brain(
    agent=agent,
    learning_algorithm="SFT",
    max_steps=500  # Usually needs fewer steps
)

# Train with different algorithms
for brain, name in [(brain_grpo, "GRPO"), (brain_dpo, "DPO"), (brain_sft, "SFT")]:
    print(f"Training with {name}...")
    brain.train(dataset=training_data)
    results = brain.evaluate(test_data)
    print(f"{name} Results: {results.average_score}")`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* GRPO & DPO Comparison */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">GRPO vs DPO: Key Differences</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-6">
                <h3 className="text-2xl font-semibold text-blue-400 mb-4">🎯 GRPO</h3>
                <p className="text-blue-200 mb-4">
                  <strong>Generalized Reinforcement Policy Optimization</strong> - An advanced RL algorithm that 
                  optimizes policies through reward-based feedback.
                </p>
                
                <h4 className="font-semibold text-blue-300 mb-2">Best For:</h4>
                <ul className="text-blue-200 text-sm space-y-1 mb-4">
                  <li>• Complex multi-step reasoning tasks</li>
                  <li>• Tool usage and API interaction</li>
                  <li>• Environments with sparse rewards</li>
                  <li>• Tasks requiring exploration</li>
                </ul>
                
                <h4 className="font-semibold text-blue-300 mb-2">Characteristics:</h4>
                <ul className="text-blue-200 text-sm space-y-1">
                  <li>• Learns from reward signals</li>
                  <li>• Handles credit assignment well</li>
                  <li>• Good at exploration-exploitation balance</li>
                  <li>• Robust to reward function changes</li>
                </ul>
              </div>
              
              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-6">
                <h3 className="text-2xl font-semibold text-purple-400 mb-4">🔄 DPO</h3>
                <p className="text-purple-200 mb-4">
                  <strong>Direct Preference Optimization</strong> - Learns directly from preference comparisons 
                  without needing explicit reward functions.
                </p>
                
                <h4 className="font-semibold text-purple-300 mb-2">Best For:</h4>
                <ul className="text-purple-200 text-sm space-y-1 mb-4">
                  <li>• Subjective quality assessment</li>
                  <li>• Human preference alignment</li>
                  <li>• Tasks with ranking-based feedback</li>
                  <li>• Creative or open-ended tasks</li>
                </ul>
                
                <h4 className="font-semibold text-purple-300 mb-2">Characteristics:</h4>
                <ul className="text-purple-200 text-sm space-y-1">
                  <li>• Learns from pairwise comparisons</li>
                  <li>• No explicit reward modeling needed</li>
                  <li>• Better for preference alignment</li>
                  <li>• More stable training dynamics</li>
                </ul>
              </div>
            </div>

            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-400 mb-2">🤔 When to Choose Which?</h4>
              <div className="grid md:grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-blue-200 mb-2"><strong>Choose GRPO when:</strong></p>
                  <ul className="text-gray-300 space-y-1">
                    <li>• You have clear success metrics</li>
                    <li>• Task involves tool/API usage</li>
                    <li>• Multi-step reasoning is required</li>
                    <li>• Environment provides natural rewards</li>
                  </ul>
                </div>
                <div>
                  <p className="text-purple-200 mb-2"><strong>Choose DPO when:</strong></p>
                  <ul className="text-gray-300 space-y-1">
                    <li>• Quality is subjective</li>
                    <li>• You have preference data</li>
                    <li>• Using LLM-as-a-judge rewards</li>
                    <li>• Human alignment is critical</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Supervised Fine-Tuning */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Supervised Fine-Tuning (SFT)</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Supervised Fine-Tuning plays a crucial role in ToolBrain, especially for knowledge distillation and 
              learning from expert demonstrations.
            </p>

            <div className="bg-green-900/20 border border-green-600 rounded-lg p-4 mb-6">
              <h3 className="text-xl font-semibold text-green-400 mb-3">🎯 Role in ToolBrain</h3>
              <ul className="text-green-200 text-sm space-y-2">
                <li>• <strong>Knowledge Distillation:</strong> Transfer knowledge from teacher models to student agents</li>
                <li>• <strong>Pre-training Phase:</strong> Prepare agents with basic capabilities before RL training</li>
                <li>• <strong>Expert Demonstrations:</strong> Learn from high-quality human or AI-generated examples</li>
                <li>• <strong>Quick Adaptation:</strong> Rapidly adapt to new domains with limited data</li>
              </ul>
            </div>

            <CodeBlock language="python">
{`# SFT for knowledge distillation workflow
def distillation_workflow():
    """Two-phase training: SFT then RL."""
    
    # Phase 1: Supervised fine-tuning from teacher
    teacher_brain = Brain(
        agent=teacher_agent,  # Large, powerful model
        learning_algorithm="SFT"
    )
    
    # Generate high-quality demonstrations
    demonstrations = teacher_brain.generate_training_examples(
        task_description="Master finance APIs",
        num_examples=1000
    )
    
    # Phase 2: Train student with SFT on demonstrations
    student_brain = Brain(
        agent=student_agent,  # Smaller, efficient model
        learning_algorithm="SFT",
        max_steps=500
    )
    
    # Distill knowledge from teacher to student
    student_brain.train(dataset=demonstrations)
    
    # Phase 3: Further improve with RL
    student_brain.learning_algorithm = "GRPO"
    student_brain.reward_func = domain_specific_reward
    
    # Continue training with reinforcement learning
    rl_data = student_brain.generate_training_examples(
        task_description="Advanced finance analysis"
    )
    student_brain.train(dataset=rl_data)
    
    return student_brain

# Usage
trained_agent = distillation_workflow()`}
            </CodeBlock>

            <div className="mt-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">SFT Configuration Options</h3>
              <CodeBlock language="python">
{`# Different SFT configurations for different purposes
configs = {
    'knowledge_distillation': {
        'learning_algorithm': 'SFT',
        'learning_rate': 2e-5,
        'max_steps': 1000,
        'batch_size': 16,
        'warmup_steps': 100
    },
    
    'rapid_adaptation': {
        'learning_algorithm': 'SFT', 
        'learning_rate': 5e-5,
        'max_steps': 200,
        'batch_size': 8,
        'warmup_steps': 20
    },
    
    'expert_imitation': {
        'learning_algorithm': 'SFT',
        'learning_rate': 1e-5,
        'max_steps': 2000,
        'batch_size': 32,
        'warmup_steps': 200
    }
}

# Apply configuration
brain = Brain(
    agent=student_agent,
    **configs['knowledge_distillation']
)`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* Algorithm Selection Guide */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Algorithm Selection Guide</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              Here's a practical guide to help you choose the right algorithm for your specific use case:
            </p>

            <div className="space-y-6">
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-blue-400 mb-3">📊 Finance & Trading Agents</h3>
                <div className="grid md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-blue-200 font-medium mb-1">Data Analysis:</p>
                    <p className="text-gray-300">SFT → GRPO</p>
                  </div>
                  <div>
                    <p className="text-blue-200 font-medium mb-1">Risk Assessment:</p>
                    <p className="text-gray-300">DPO (subjective judgment)</p>
                  </div>
                  <div>
                    <p className="text-blue-200 font-medium mb-1">Trading Strategies:</p>
                    <p className="text-gray-300">GRPO (reward-based)</p>
                  </div>
                </div>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-purple-400 mb-3">💻 Code Generation Agents</h3>
                <div className="grid md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-purple-200 font-medium mb-1">Basic Coding:</p>
                    <p className="text-gray-300">SFT (from examples)</p>
                  </div>
                  <div>
                    <p className="text-purple-200 font-medium mb-1">Code Quality:</p>
                    <p className="text-gray-300">DPO (preference-based)</p>
                  </div>
                  <div>
                    <p className="text-purple-200 font-medium mb-1">Complex Problems:</p>
                    <p className="text-gray-300">GRPO (multi-step)</p>
                  </div>
                </div>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-green-400 mb-3">🔍 Research & Analysis Agents</h3>
                <div className="grid md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <p className="text-green-200 font-medium mb-1">Information Gathering:</p>
                    <p className="text-gray-300">GRPO (tool usage)</p>
                  </div>
                  <div>
                    <p className="text-green-200 font-medium mb-1">Quality Assessment:</p>
                    <p className="text-gray-300">DPO (subjective)</p>
                  </div>
                  <div>
                    <p className="text-green-200 font-medium mb-1">Report Writing:</p>
                    <p className="text-gray-300">SFT → DPO</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Practical Examples */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Practical Examples</h2>
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">Multi-Algorithm Training Pipeline</h3>
                <CodeBlock language="python">
{`def multi_stage_training(agent, task_description):
    """Advanced training pipeline using multiple algorithms."""
    
    # Stage 1: Knowledge transfer with SFT
    print("Stage 1: Knowledge Distillation...")
    sft_brain = Brain(
        agent=agent,
        learning_algorithm="SFT",
        max_steps=500
    )
    
    expert_demos = sft_brain.generate_training_examples(
        task_description=f"Basic {task_description}"
    )
    sft_brain.distill(dataset=expert_demos, teacher_model_id="GPT-4-Turbo")
    sft_brain.train(dataset=expert_demos)
    
    # Stage 2: Preference alignment with DPO
    print("Stage 2: Preference Alignment...")
    dpo_brain = Brain(
        agent=agent,  # Same agent, now pre-trained
        learning_algorithm="DPO", 
        reward_func=reward_llm_judge_via_ranking,
        max_steps=300
    )
    
    preference_data = dpo_brain.generate_training_examples(
        task_description=f"High-quality {task_description}"
    )
    dpo_brain.train(dataset=preference_data)
    
    # Stage 3: Task optimization with GRPO
    print("Stage 3: Task Optimization...")
    grpo_brain = Brain(
        agent=agent,  # Now aligned and knowledgeable
        learning_algorithm="GRPO",
        reward_func=task_specific_reward,
        max_steps=800
    )
    
    optimization_data = grpo_brain.generate_training_examples(
        task_description=f"Expert {task_description}"
    )
    grpo_brain.train(dataset=optimization_data)
    
    return agent

# Example usage
finance_agent = CodeAgent(model="Qwen/3B-Instruct", tools=finance_tools)
trained_agent = multi_stage_training(finance_agent, "financial analysis")`}
                </CodeBlock>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-green-400 mb-3">Algorithm Comparison Experiment</h3>
                <CodeBlock language="python">
{`def compare_algorithms(agent, dataset, algorithms=["SFT", "DPO", "GRPO"]):
    """Compare different algorithms on the same task."""
    
    results = {}
    
    for algorithm in algorithms:
        print(f"Training with {algorithm}...")
        
        brain = Brain(
            agent=agent.copy(),  # Fresh copy for each algorithm
            learning_algorithm=algorithm,
            reward_func=get_reward_for_algorithm(algorithm),
            max_steps=1000
        )
        
        # Train and evaluate
        brain.train(dataset=dataset)
        evaluation = brain.evaluate(test_dataset)
        
        results[algorithm] = {
            'accuracy': evaluation.accuracy,
            'efficiency': evaluation.efficiency,
            'quality': evaluation.quality_score,
            'training_time': evaluation.training_time
        }
    
    # Print comparison
    print("\\nAlgorithm Comparison Results:")
    for alg, metrics in results.items():
        print(f"{alg:>6}: Acc={metrics['accuracy']:.3f}, "
              f"Eff={metrics['efficiency']:.3f}, "
              f"Qual={metrics['quality']:.3f}")
    
    return results

def get_reward_for_algorithm(algorithm):
    """Select appropriate reward function for algorithm."""
    if algorithm == "DPO":
        return reward_llm_judge_via_ranking
    elif algorithm == "GRPO": 
        return task_specific_reward
    else:  # SFT
        return None  # Uses supervised loss`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}