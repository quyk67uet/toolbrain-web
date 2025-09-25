'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function FlexibleRewards() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Guide: Flexible Rewards</h1>
          <p className="text-gray-300 text-lg">
            Discover ToolBrain's powerful reward system that supports both user-defined functions and LLM-as-a-judge approaches for training intelligent agents.
          </p>
        </div>

        {/* Overview */}
        <section className="mb-12">
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Effective reward engineering is crucial for training high-quality agents. ToolBrain provides a flexible reward system 
              that accommodates different evaluation approaches, from simple rule-based functions to sophisticated LLM-based judgments.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-blue-400 mb-3">🎯 User-Defined Functions</h3>
                <p className="text-blue-200 text-sm">
                  Create custom reward functions based on domain expertise and specific task requirements. 
                  Perfect for well-defined metrics and performance criteria.
                </p>
              </div>
              
              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-purple-400 mb-3">🤖 LLM-as-a-Judge</h3>
                <p className="text-purple-200 text-sm">
                  Leverage large language models to evaluate agent performance using natural language criteria. 
                  Ideal for complex, subjective tasks where rules are hard to define.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* User-Defined Functions */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">User-Defined Reward Functions</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              ToolBrain comes with built-in reward functions that you can use directly or customize for your specific needs. 
              Here's an example from <code className="bg-gray-700 px-2 py-1 rounded">toolbrain/rewards.py</code>:
            </p>
            
            <CodeBlock language="python">
{`def reward_step_efficiency(trajectory, **kwargs):
    """
    Reward function that evaluates agent efficiency based on the number of steps
    taken to complete a task. Encourages agents to solve problems efficiently.
    
    Args:
        trajectory: The agent's execution trajectory containing steps and outcomes
        **kwargs: Additional parameters for customization
        
    Returns:
        float: Reward score between 0.0 and 1.0
    """
    if not trajectory or not trajectory.steps:
        return 0.0
    
    # Get the number of steps taken
    num_steps = len(trajectory.steps)
    
    # Check if the task was completed successfully
    success = trajectory.final_outcome and trajectory.final_outcome.success
    
    if not success:
        # Penalize failed attempts
        return 0.0
    
    # Define efficiency thresholds
    optimal_steps = kwargs.get('optimal_steps', 3)
    max_acceptable_steps = kwargs.get('max_steps', 10)
    
    if num_steps <= optimal_steps:
        # Perfect efficiency
        return 1.0
    elif num_steps <= max_acceptable_steps:
        # Decreasing reward based on excess steps
        efficiency_ratio = (max_acceptable_steps - num_steps) / (max_acceptable_steps - optimal_steps)
        return max(0.1, efficiency_ratio)
    else:
        # Too many steps, minimal reward
        return 0.1
        
# Additional reward components that can be combined
def reward_code_quality(trajectory, **kwargs):
    """Evaluate code quality in generated solutions."""
    if not trajectory.final_code:
        return 0.0
        
    quality_score = 0.0
    
    # Check for proper error handling
    if 'try:' in trajectory.final_code and 'except:' in trajectory.final_code:
        quality_score += 0.3
    
    # Check for documentation
    if '"""' in trajectory.final_code or "'''" in trajectory.final_code:
        quality_score += 0.2
        
    # Check for type hints
    if '->' in trajectory.final_code or ': ' in trajectory.final_code:
        quality_score += 0.2
        
    # Check for meaningful variable names
    if not any(name in trajectory.final_code for name in ['x', 'y', 'temp', 'var']):
        quality_score += 0.3
    
    return min(1.0, quality_score)

def reward_tool_usage_correctness(trajectory, **kwargs):
    """Evaluate correct usage of tools and APIs."""
    if not trajectory.tool_calls:
        return 0.5  # Neutral if no tools were needed
    
    correct_calls = 0
    total_calls = len(trajectory.tool_calls)
    
    for call in trajectory.tool_calls:
        # Check if tool was used appropriately
        if call.success and call.parameters_valid:
            correct_calls += 1
    
    return correct_calls / total_calls if total_calls > 0 else 0.0`}
            </CodeBlock>

            <div className="mt-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Combining Multiple Reward Components</h3>
              <p className="text-gray-300 mb-4">
                You can combine multiple reward functions to create comprehensive evaluation criteria:
              </p>
              
              <CodeBlock language="python">
{`def create_composite_reward(weights=None):
    """
    Create a composite reward function from multiple components.
    
    Args:
        weights: Dictionary mapping reward function names to weights
    """
    if weights is None:
        weights = {
            'efficiency': 0.4,
            'code_quality': 0.3,
            'tool_usage': 0.3
        }
    
    def composite_reward(trajectory, **kwargs):
        scores = {}
        
        # Calculate individual reward components
        scores['efficiency'] = reward_step_efficiency(trajectory, **kwargs)
        scores['code_quality'] = reward_code_quality(trajectory, **kwargs)
        scores['tool_usage'] = reward_tool_usage_correctness(trajectory, **kwargs)
        
        # Weighted combination
        total_score = sum(scores[key] * weights[key] for key in weights)
        
        # Log detailed breakdown for debugging
        kwargs.setdefault('logging', {})['reward_breakdown'] = scores
        
        return total_score
    
    return composite_reward

# Usage example
reward_func = create_composite_reward(weights={
    'efficiency': 0.5,      # Prioritize efficiency
    'code_quality': 0.3,    # Good code practices
    'tool_usage': 0.2       # Correct tool usage
})`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* LLM-as-a-Judge */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">LLM-as-a-Judge via Ranking</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              For complex tasks where defining explicit rules is challenging, ToolBrain supports using large language models 
              as judges. The key insight is that <strong>"ranking is better than scoring"</strong> - LLMs are more reliable 
              at comparing solutions than assigning absolute scores.
            </p>

            <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-4 mb-6">
              <h3 className="text-xl font-semibold text-yellow-400 mb-3">🧠 Why Ranking &gt; Scoring?</h3>
              <ul className="text-yellow-200 text-sm space-y-2">
                <li>• <strong>Consistency:</strong> LLMs are more consistent when comparing A vs B than scoring A and B separately</li>
                <li>• <strong>Calibration:</strong> Eliminates issues with score scale interpretation</li>
                <li>• <strong>Reliability:</strong> Reduces variance in evaluation across different prompts</li>
                <li>• <strong>Alignment:</strong> Better captures human preferences and nuanced quality judgments</li>
              </ul>
            </div>

            <h3 className="text-xl font-semibold text-blue-400 mb-3">Simple API Usage</h3>
            <p className="text-gray-300 mb-4">
              Using LLM-as-a-judge is as simple as setting the reward function:
            </p>
            
            <CodeBlock language="python">
{`from toolbrain.rewards import reward_llm_judge_via_ranking

# Initialize Brain with LLM-based reward
brain = Brain(
    agent=your_agent,
    reward_func=reward_llm_judge_via_ranking,  # Simple one-liner!
    learning_algorithm="GRPO"
)

# The LLM judge will automatically:
# 1. Compare agent outputs pairwise
# 2. Rank solutions based on quality
# 3. Convert rankings to reward signals
# 4. Provide detailed feedback for training`}
            </CodeBlock>

            <div className="mt-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">How LLM Ranking Works</h3>
              <p className="text-gray-300 mb-4">
                Behind the scenes, the LLM judge implements a sophisticated ranking process:
              </p>
              
              <CodeBlock language="python">
{`def reward_llm_judge_via_ranking(trajectory, **kwargs):
    """
    LLM-based reward using pairwise ranking approach.
    
    The judge compares the current trajectory against reference solutions
    and provides relative ranking-based rewards.
    """
    
    # 1. Collect comparison candidates
    current_solution = trajectory.final_output
    reference_solutions = kwargs.get('reference_solutions', [])
    
    # 2. Perform pairwise comparisons
    ranking_prompt = f"""
    Compare the following two solutions to the task:
    
    Task: {trajectory.task_description}
    
    Solution A: {current_solution}
    Solution B: {{reference_solution}}
    
    Which solution is better? Consider:
    - Correctness and accuracy
    - Efficiency and elegance
    - Clarity and readability
    - Proper tool usage
    
    Respond with: "A is better", "B is better", or "Equal quality"
    Provide a brief explanation.
    """
    
    # 3. Calculate relative ranking score
    wins = 0
    total_comparisons = 0
    
    for ref_solution in reference_solutions:
        comparison_result = llm_judge.compare(
            prompt=ranking_prompt.format(reference_solution=ref_solution)
        )
        
        if "A is better" in comparison_result:
            wins += 1
        elif "Equal quality" in comparison_result:
            wins += 0.5
        
        total_comparisons += 1
    
    # 4. Convert ranking to reward signal
    if total_comparisons == 0:
        return 0.5  # Neutral if no comparisons
    
    ranking_score = wins / total_comparisons
    return ranking_score`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* Custom Judge Configuration */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Custom Judge Configuration</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              You can customize the LLM judge with specific criteria and evaluation prompts:
            </p>
            
            <CodeBlock language="python">
{`from toolbrain.rewards import create_custom_llm_judge

# Create custom judge with specific criteria
custom_judge = create_custom_llm_judge(
    judge_model="GPT-4-Turbo",
    evaluation_criteria=[
        "Technical correctness",
        "Code efficiency and elegance", 
        "Proper error handling",
        "Documentation quality",
        "Security best practices"
    ],
    comparison_style="detailed",  # or "quick", "thorough"
    temperature=0.1,  # Low temperature for consistent judgments
    max_comparisons=5  # Number of reference solutions to compare against
)

# Use in Brain initialization
brain = Brain(
    agent=code_agent,
    reward_func=custom_judge,
    learning_algorithm="DPO"  # DPO works well with ranking-based rewards
)

# Alternative: Hybrid approach combining both methods
def hybrid_reward(trajectory, **kwargs):
    """Combine rule-based and LLM-based evaluation."""
    
    # Get rule-based score
    rule_score = reward_step_efficiency(trajectory, **kwargs)
    
    # Get LLM-based score (only for high-stakes decisions)
    if trajectory.complexity_score > 0.7:
        llm_score = reward_llm_judge_via_ranking(trajectory, **kwargs)
        # Weighted combination
        return 0.3 * rule_score + 0.7 * llm_score
    else:
        # Use rule-based for simple cases
        return rule_score

brain = Brain(
    agent=your_agent,
    reward_func=hybrid_reward,
    learning_algorithm="GRPO"
)`}
            </CodeBlock>

            <div className="mt-6 p-4 bg-green-900/20 border border-green-600 rounded-lg">
              <h4 className="font-semibold text-green-400 mb-2">🎯 Best Practices</h4>
              <ul className="text-green-200 text-sm space-y-1">
                <li>• Start with simple rule-based rewards for well-defined tasks</li>
                <li>• Use LLM judges for subjective or complex evaluation criteria</li>
                <li>• Combine multiple reward components for comprehensive evaluation</li>
                <li>• Always validate reward functions with manual spot-checks</li>
                <li>• Use ranking-based LLM evaluation for more reliable results</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Real-World Examples */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Real-World Examples</h2>
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">Finance Agent Reward</h3>
                <CodeBlock language="python">
{`def finance_agent_reward(trajectory, **kwargs):
    """Reward for financial analysis tasks."""
    scores = {}
    
    # Data accuracy (rule-based)
    scores['accuracy'] = check_calculation_accuracy(
        trajectory.calculations
    )
    
    # Risk assessment quality (LLM judge)
    if trajectory.risk_analysis:
        scores['risk_quality'] = reward_llm_judge_via_ranking(
            trajectory, 
            focus="risk assessment quality"
        )
    
    # Compliance with regulations (rule-based)
    scores['compliance'] = check_regulatory_compliance(
        trajectory.recommendations
    )
    
    return weighted_average(scores, {
        'accuracy': 0.5,
        'risk_quality': 0.3,
        'compliance': 0.2
    })`}
                </CodeBlock>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-purple-400 mb-3">Code Review Agent Reward</h3>
                <CodeBlock language="python">
{`def code_review_reward(trajectory, **kwargs):
    """Reward for code review tasks."""
    
    # Combine multiple evaluation approaches
    review_quality = reward_llm_judge_via_ranking(
        trajectory,
        evaluation_criteria=[
            "Identifies security vulnerabilities",
            "Suggests performance improvements", 
            "Maintains code style consistency",
            "Provides constructive feedback"
        ]
    )
    
    # Check for objective metrics
    completeness = len(trajectory.identified_issues) / \
                  len(kwargs['ground_truth_issues'])
    
    return 0.7 * review_quality + 0.3 * completeness`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}