'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function Quickstart() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Quickstart: Your First Trained Agent in 7 Lines of Code</h1>
          <p className="text-gray-300 text-lg">
            Guide through the simplest possible end-to-end example. This should be your "Aha!" moment with ToolBrain.
          </p>
        </div>

        {/* Introduction */}
        <section className="mb-12">
          <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-6 mb-8">
            <h2 className="text-2xl font-semibold text-blue-400 mb-4">🎯 Goal</h2>
            <p className="text-blue-200">
              In this quickstart, we&apos;ll train an intelligent agent to master finance APIs through reinforcement learning. 
              The agent will learn to use tools effectively by practicing on generated examples and receiving feedback 
              from a teacher model.
            </p>
          </div>
        </section>

        {/* Step 1: Define Tool & Reward */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Step 1: Define a Tool & Reward Function</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              First, let&apos;s set up a simple tool and reward function. For this example, we&apos;ll use finance APIs and 
              define how to judge the agent&apos;s performance.
            </p>
            
            <div className="mb-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Available Tools</h3>
              <CodeBlock language="python">
{`# Example finance tools the agent will learn to use
finance_tools = [
    get_stock_price,      # Get current stock price
    calculate_portfolio,  # Calculate portfolio value  
    get_market_data,     # Fetch market indicators
    risk_analysis        # Perform risk assessment
]`}
              </CodeBlock>
            </div>

            <div>
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Reward Function</h3>
              <CodeBlock language="python">
{`# Define how to evaluate the agent's performance
reward_func = llm_judge_or_user_defined  # Uses LLM or custom logic to judge quality`}
              </CodeBlock>
              <p className="text-gray-400 text-sm mt-2">
                The reward function evaluates how well the agent uses tools to complete financial analysis tasks.
              </p>
            </div>
          </div>
        </section>

        {/* Step 2: The 7-Line Workflow */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Step 2: The 7-Line ToolBrain Workflow</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              Here&apos;s the complete ToolBrain workflow in just 7 lines of code. Each step builds upon the previous one 
              to create a fully trained agent.
            </p>

            <div className="space-y-8">
              {/* Line 1-2: Create Student Agent */}
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">1. Create the Student Agent</h3>
                <CodeBlock language="python">
{`student_agent = CodeAgent(
    model="Qwen/3B-Instruct",
    tools=[all_available_tools]
)`}
                </CodeBlock>
                <p className="text-gray-400 text-sm mt-2">
                  Initialize a code-generating agent with a 3B parameter model and provide access to all available tools.
                </p>
              </div>

              {/* Line 3-6: Configure Brain */}
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">2. Configure the Training Brain</h3>
                <CodeBlock language="python">
{`brain = Brain(
    agent=student_agent, 
    reward_func=reward_func,
    enable_tool_retrieval=True,
    learning_algorithm="GRPO"
)`}
                </CodeBlock>
                <p className="text-gray-400 text-sm mt-2">
                  Set up the Brain with your agent, reward function, tool retrieval enabled, and GRPO (Generalized Reinforcement Policy Optimization) algorithm.
                </p>
              </div>

              {/* Line 7: Generate Training Examples */}
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">3. Generate Training Data</h3>
                <CodeBlock language="python">
{`training_tasks = brain.generate_training_examples(
    task_description="Master the finance APIs"
)`}
                </CodeBlock>
                <p className="text-gray-400 text-sm mt-2">
                  Automatically generate diverse training examples focused on mastering finance API usage.
                </p>
              </div>

              {/* Line 8: Distill Knowledge */}
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">4. Knowledge Distillation</h3>
                <CodeBlock language="python">
{`brain.distill(
    dataset=training_tasks,
    teacher_model_id="GPT-4-Turbo"
)`}
                </CodeBlock>
                <p className="text-gray-400 text-sm mt-2">
                  Use a powerful teacher model (GPT-4-Turbo) to provide high-quality demonstrations for the student agent.
                </p>
              </div>

              {/* Line 9: Train */}
              <div>
                <h3 className="text-lg font-semibold text-blue-400 mb-3">5. Start Training</h3>
                <CodeBlock language="python">
{`brain.train(dataset=training_tasks)`}
                </CodeBlock>
                <p className="text-gray-400 text-sm mt-2">
                  Begin the reinforcement learning training process using the generated dataset.
                </p>
              </div>
            </div>

            {/* Complete Code Block */}
            <div className="mt-8 p-4 bg-gray-900 rounded-lg border border-green-600">
              <h3 className="text-lg font-semibold text-green-400 mb-3">🚀 Complete 7-Line Script</h3>
              <CodeBlock language="python">
{`reward_func = llm_judge_or_user_defined

student_agent = CodeAgent(
    model="Qwen/3B-Instruct",
    tools=[all_available_tools]
)

brain = Brain(
    agent=student_agent, 
    reward_func=reward_func,
    enable_tool_retrieval=True,
    learning_algorithm="GRPO"
)

training_tasks = brain.generate_training_examples(
    task_description="Master the finance APIs"
)

brain.distill(
    dataset=training_tasks,
    teacher_model_id="GPT-4-Turbo"
)

brain.train(dataset=training_tasks)`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* Step 3: Run it */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Step 3: Run It!</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Save the code above as <code className="bg-gray-700 px-2 py-1 rounded">quickstart.py</code> and run it:
            </p>
            
            <CodeBlock language="bash">
{`python quickstart.py`}
            </CodeBlock>
            
            <div className="mt-4 p-4 bg-yellow-900/20 border border-yellow-600 rounded-lg">
              <h4 className="font-semibold text-yellow-400 mb-2">⚡ What Happens Next</h4>
              <ul className="text-yellow-200 text-sm space-y-1">
                <li>• ToolBrain generates diverse finance API tasks</li>
                <li>• GPT-4-Turbo provides expert demonstrations</li>
                <li>• Your student agent learns through reinforcement learning</li>
                <li>• Training progress is monitored and optimized automatically</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Step 4: The Result */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Step 4: The Result</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              After training completes, you&apos;ll have an intelligent agent that has learned to effectively use finance APIs. 
              Here&apos;s what you can expect:
            </p>

            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-green-400 mb-2">✅ Agent Capabilities</h3>
                <ul className="text-green-200 text-sm space-y-1">
                  <li>• Correctly uses finance APIs</li>
                  <li>• Handles complex multi-step queries</li>
                  <li>• Provides accurate financial analysis</li>
                  <li>• Adapts to new scenarios</li>
                </ul>
              </div>
              
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-blue-400 mb-2">📊 Training Metrics</h3>
                <ul className="text-blue-200 text-sm space-y-1">
                  <li>• Reward progression over time</li>
                  <li>• Tool usage accuracy</li>
                  <li>• Task completion rate</li>
                  <li>• Performance comparisons</li>
                </ul>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Testing Your Trained Agent</h3>
              <CodeBlock language="python">
{`# Test the trained agent on a new task
result = student_agent.execute(
    task="Analyze the risk profile of AAPL stock and suggest portfolio allocation"
)

print(f"Agent response: {result.response}")
print(f"Tools used: {result.tools_used}")
print(f"Execution trace: {result.trace}")`}
              </CodeBlock>
            </div>

            <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-4">
              <h4 className="font-semibold text-purple-400 mb-2">🎉 Congratulations!</h4>
              <p className="text-purple-200 text-sm">
                You&apos;ve just trained your first intelligent agent with ToolBrain! The agent can now autonomously 
                use finance APIs to solve complex tasks. From here, you can explore advanced features like 
                multi-agent systems, custom reward functions, and distributed training.
              </p>
            </div>
          </div>
        </section>

        {/* Next Steps */}
        <section className="mb-12">
          <div className="bg-gray-800 rounded-lg p-6 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">Ready for More?</h2>
            <p className="text-gray-300 mb-6">
              Now that you&apos;ve seen the power of ToolBrain, explore advanced tutorials and features.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-6 py-3 rounded-lg font-medium transition-colors duration-200">
                Advanced Tutorials
              </button>
              <button className="border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] px-6 py-3 rounded-lg font-medium transition-colors duration-200">
                Explore Key Features
              </button>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}