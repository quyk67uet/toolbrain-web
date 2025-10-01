'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function Quickstart() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-[#E6EDF3] mb-6">Quickstart: Your First Trained Agent</h1>
          <p className="text-xl text-gray-400 leading-relaxed mb-8">
            In this guide, we&apos;ll walk you through the entire process of training your first agent with ToolBrain. 
            We&apos;ll train a simple CodeAgent to reliably use a multiply tool. By the end, you&apos;ll have a 
            fine-tuned agent saved to your disk, all in just a few simple steps.
          </p>

          {/* Prerequisites */}
          <div className="bg-[#2D1B1B] border border-[#F85149]/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-[#F85149] mb-3">⚡ Prerequisites</h3>
            <p className="text-gray-300 mb-4">
              Before you begin, make sure you have installed ToolBrain with the necessary dependencies for training. 
              We recommend the <code className="bg-[#21262D] px-2 py-1 rounded text-[#58A6FF]">[unsloth]</code> extra for the best performance.
            </p>
            <CodeBlock language="bash">
              pip install &quot;toolbrain[unsloth]&quot;
            </CodeBlock>
          </div>
        </div>

        {/* Step 1: Define Your Agent and Task */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">Step 1: Define Your Agent and Task</h2>
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            First, let&apos;s set up the basic building blocks. We&apos;ll create a standard smolagents.CodeAgent and 
            provide it with a simple multiply tool. ToolBrain is designed to work with your existing agent 
            definitions without modification. Then, we&apos;ll define the task we want the agent to learn.
          </p>

          {/* Code Block 1.1: The Tool & Agent */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-[#58A6FF] mb-4">1.1. The Tool & Agent</h3>
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <CodeBlock language="python">
{`from smolagents import CodeAgent, tool
from toolbrain.models import UnslothModel # Using our optimized model class

# 1. Define a tool for the agent
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

# 2. Create your agent
my_agent = CodeAgent(
    model=UnslothModel(model_id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=[multiply]
)`}
              </CodeBlock>
            </div>
          </div>

          {/* Code Block 1.2: The Task */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-[#58A6FF] mb-4">1.2. The Task</h3>
            <p className="text-gray-400 mb-4">
              Our training dataset will consist of a single task: asking the agent to calculate 6 multiplied by 7, 
              where the correct answer is 42.
            </p>
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <CodeBlock language="python">
{`# 3. Define the training task
dataset = [{
    "query": "What is 6 multiplied by 7?",
    "gold_answer": "42" # This will be used by the reward function
}]`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* Step 2: Train with the Brain */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">Step 2: Train with the Brain</h2>
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            This is where the magic happens. We import the Brain and one of its built-in reward functions, 
            reward_exact_match. We then initialize the Brain with our components and start the entire RL 
            training process with a single <code className="bg-[#21262D] px-2 py-1 rounded text-[#58A6FF]">.train()</code> command.
          </p>

          {/* Code Block 2.1: Initialize and Train */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-[#58A6FF] mb-4">2.1. Initialize and Train</h3>
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <CodeBlock language="python">
{`from toolbrain import Brain
from toolbrain.rewards import reward_exact_match # Import a built-in reward

# 1. Initialize the Brain
brain = Brain(
    agent=my_agent,
    # Use a built-in function that checks if the agent's
    # final_answer matches the 'gold_answer' in our dataset.
    reward_func=reward_exact_match,
    learning_algorithm="GRPO",
    num_group_members=4 # Generate 4 attempts per query for robust learning
)

# 2. Run the training!
brain.train(dataset=dataset, num_iterations=10)`}
              </CodeBlock>
            </div>
            <div className="mt-4 bg-[#1B2D1B] border border-[#3FB950]/30 rounded-lg p-4">
              <p className="text-[#3FB950] text-sm">
                <strong>💡 Note:</strong> After running, you will see logs of the training process, including 
                trace collection, reward computation, and model updates.
              </p>
            </div>
          </div>
        </section>

        {/* Step 3: Save and Reload Your Trained Agent */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">Step 3: Save and Reload Your Trained Agent</h2>
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            After training, ToolBrain makes it easy to save your agent&apos;s learned skills (the LoRA adapters) 
            and reload them for later use or deployment.
          </p>

          {/* Code Block 3.1: Save the Agent */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-[#58A6FF] mb-4">3.1. Save the Agent</h3>
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <CodeBlock language="python">
{`# Save the trained LoRA adapters to a directory
brain.save("./my_trained_math_agent")`}
              </CodeBlock>
            </div>
          </div>

          {/* Code Block 3.2: Load the Agent */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-[#58A6FF] mb-4">3.2. Load the Agent</h3>
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <CodeBlock language="python">
{`# You can later load these adapters into a fresh agent instance

# First, create a base agent (same as before)
base_agent = CodeAgent(
    model=UnslothModel(model_id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=[multiply]
)

# Now, load the trained skills into it
trained_agent = Brain.load_agent(
    model_dir="./my_trained_math_agent",
    agent_to_load_into=base_agent
)`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* Next Steps */}
        <section className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">Next Steps</h2>
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8">
            <p className="text-xl text-gray-300 mb-6 leading-relaxed">
              <strong className="text-[#3FB950]">Congratulations!</strong> You&apos;ve just trained, saved, and reloaded your first agent with ToolBrain.
            </p>
            <p className="text-lg text-gray-400 mb-8 leading-relaxed">
              You&apos;ve seen how the Brain API simplifies a complex RL workflow into a few intuitive steps. 
              To dive deeper into our advanced capabilities, we recommend exploring the following sections:
            </p>

            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <div className="bg-[#2D1B1B] border border-[#F85149]/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#F85149] mb-3">🔧 Key Features</h3>
                <p className="text-gray-300 text-sm">
                  Get a detailed overview of features like our LLM-as-a-Judge, Knowledge Distillation, and more.
                </p>
              </div>
              <div className="bg-[#1B2D1B] border border-[#3FB950]/30 rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#3FB950] mb-3">📚 Conceptual Guides</h3>
                <p className="text-gray-300 text-sm">
                  Understand the core design principles behind ToolBrain, like the &apos;Coach-Athlete&apos; paradigm.
                </p>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <a 
                href="/key-features"
                className="bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-8 py-3 rounded-lg font-medium transition-colors duration-200 text-center"
              >
                Explore Key Features
              </a>
              <a 
                href="/conceptual-guides"
                className="border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] px-8 py-3 rounded-lg font-medium transition-colors duration-200 text-center"
              >
                Read Conceptual Guides
              </a>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}