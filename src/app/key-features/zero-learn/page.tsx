'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function ZeroLearnTaskGeneration() {
  return (
    <Layout>
      <div className="max-w-6xl mx-auto px-4 py-16">
        
        {/* PHẦN 1: THE PROBLEM - "THE BLANK PAGE" */}
        <div className="mb-20">
          <h1 className="text-5xl font-bold text-[#E6EDF3] mb-8 text-center">
            Zero-Learn Task Generation
          </h1>
          
          <div className="max-w-4xl mx-auto text-center mb-12">
            <p className="text-xl text-gray-400 mb-6 leading-relaxed">
              One of the biggest hurdles in training an agent is the lack of high-quality, tool-specific training data. 
              Manually writing dozens or hundreds of diverse and realistic task examples is a time-consuming and tedious process.
            </p>
            <p className="text-xl text-gray-300 leading-relaxed">
              ToolBrain solves this <span className="text-[#58A6FF] font-semibold">"blank page"</span> problem with its 
              <code className="bg-[#161B22] px-2 py-1 rounded text-[#58A6FF] mx-2">generate_training_examples</code> 
              method, allowing you to bootstrap the entire training process from a simple, high-level description.
            </p>
          </div>

          {/* Problem Illustration */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8">
            <div className="text-center">
              <div className="text-6xl mb-4">📄</div>
              <h3 className="text-2xl font-bold text-[#58A6FF] mb-4">The "Blank Page" Challenge</h3>
              <p className="text-gray-400 max-w-2xl mx-auto">
                Starting with zero training examples means spending countless hours manually crafting realistic, 
                diverse tasks that properly exercise your agent's tools.
              </p>
            </div>
          </div>
        </div>

        {/* PHẦN 2: THE SOLUTION - AUTOMATED TASK CREATION */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] text-center mb-8">
            How It Works: From Description to Dataset
          </h2>
          
          <p className="text-lg text-gray-400 mb-12 text-center max-w-3xl mx-auto leading-relaxed">
            The <code className="bg-[#161B22] px-2 py-1 rounded text-[#58A6FF]">generate_training_examples</code> method 
            uses a powerful LLM to act as a <span className="text-[#3FB950] font-semibold">"curriculum designer"</span>. 
            You provide it with a high-level description of the desired tasks and the tools your agent has. 
            It then generates a list of diverse, realistic, and tool-aligned query strings that can be used directly as a training dataset.
          </p>

          {/* VISUAL PIPELINE */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12">
            
            {/* Desktop Layout */}
            <div className="hidden lg:flex items-center justify-center space-x-8">
              
              {/* Input */}
              <div className="flex-1 max-w-xs">
                <div className="bg-gradient-to-br from-[#58A6FF]/20 to-[#4A90E2]/20 border border-[#58A6FF]/50 rounded-xl p-6 h-full">
                  <div className="text-center">
                    <div className="text-4xl mb-3">📝</div>
                    <h3 className="text-lg font-bold text-[#58A6FF] mb-2">Task Description</h3>
                    <p className="text-xs text-gray-400 mb-3">High-level requirements</p>
                    <div className="border-t border-[#58A6FF]/20 pt-3">
                      <div className="text-2xl mb-1">🛠️</div>
                      <p className="text-xs text-gray-400">Available Tools</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex-shrink-0">
                <div className="text-[#58A6FF] text-4xl">→</div>
              </div>

              {/* Process */}
              <div className="flex-1 max-w-xs">
                <div className="bg-gradient-to-br from-[#7C3AED]/20 to-[#6366F1]/20 border border-[#7C3AED]/50 rounded-xl p-6 h-full">
                  <div className="text-center">
                    <div className="text-4xl mb-3">🧠</div>
                    <h3 className="text-lg font-bold text-[#7C3AED] mb-2">LLM Curriculum</h3>
                    <h3 className="text-lg font-bold text-[#7C3AED] mb-3">Designer</h3>
                    <code className="text-xs text-[#7C3AED] bg-[#7C3AED]/10 px-2 py-1 rounded block">
                      generate_training_examples()
                    </code>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex-shrink-0">
                <div className="text-[#58A6FF] text-4xl">→</div>
              </div>

              {/* Output */}
              <div className="flex-1 max-w-xs">
                <div className="bg-gradient-to-br from-[#3FB950]/20 to-[#10B981]/20 border border-[#3FB950]/50 rounded-xl p-6 h-full">
                  <div className="text-center">
                    <div className="text-4xl mb-3">📋</div>
                    <h3 className="text-lg font-bold text-[#3FB950] mb-3">Training Dataset</h3>
                    <div className="text-xs text-gray-300 space-y-1 text-left bg-[#3FB950]/5 rounded p-2">
                      <div className="truncate">"Calculate..."</div>
                      <div className="truncate">"What is..."</div>
                      <div className="truncate">"How much..."</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Mobile Layout */}
            <div className="flex flex-col lg:hidden space-y-6">
              <div className="text-center">
                <div className="bg-gradient-to-br from-[#58A6FF]/20 to-[#4A90E2]/20 border border-[#58A6FF]/50 rounded-xl p-6 mx-auto max-w-sm">
                  <div className="text-4xl mb-3">📝</div>
                  <h3 className="text-lg font-bold text-[#58A6FF] mb-2">Task Description</h3>
                  <p className="text-xs text-gray-400 mb-3">High-level requirements</p>
                  <div className="border-t border-[#58A6FF]/20 pt-3">
                    <div className="text-2xl mb-1">🛠️</div>
                    <p className="text-xs text-gray-400">Available Tools</p>
                  </div>
                </div>
              </div>

              <div className="text-center text-[#58A6FF] text-3xl">↓</div>

              <div className="text-center">
                <div className="bg-gradient-to-br from-[#7C3AED]/20 to-[#6366F1]/20 border border-[#7C3AED]/50 rounded-xl p-6 mx-auto max-w-sm">
                  <div className="text-4xl mb-3">🧠</div>
                  <h3 className="text-lg font-bold text-[#7C3AED] mb-2">LLM Curriculum Designer</h3>
                  <code className="text-xs text-[#7C3AED] bg-[#7C3AED]/10 px-2 py-1 rounded">
                    generate_training_examples()
                  </code>
                </div>
              </div>

              <div className="text-center text-[#58A6FF] text-3xl">↓</div>

              <div className="text-center">
                <div className="bg-gradient-to-br from-[#3FB950]/20 to-[#10B981]/20 border border-[#3FB950]/50 rounded-xl p-6 mx-auto max-w-sm">
                  <div className="text-4xl mb-3">📋</div>
                  <h3 className="text-lg font-bold text-[#3FB950] mb-3">Training Dataset</h3>
                  <div className="text-xs text-gray-300 space-y-1 text-left bg-[#3FB950]/5 rounded p-2">
                    <div>"Calculate..."</div>
                    <div>"What is..."</div>
                    <div>"How much..."</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* PHẦN 3: THE API IN ACTION - A FINANCE AGENT EXAMPLE */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] text-center mb-8">
            How to Use It: Generating Tasks for a Finance Agent
          </h2>
          
          <p className="text-lg text-gray-400 mb-12 text-center max-w-3xl mx-auto leading-relaxed">
            Let's walk through an example. Imagine we want to train an agent to use a set of financial calculation tools.
          </p>

          {/* Code Example 1: Setup */}
          <div className="mb-12">
            <h3 className="text-2xl font-bold text-[#58A6FF] mb-4">Step 1: Setup</h3>
            <p className="text-gray-400 mb-4">
              First, we define our agent and its financial tools, then initialize the Brain.
            </p>
            <CodeBlock
              language="python"
            >
{`from toolbrain import Brain, create_agent
from my_finance_tools import (
    calculate_compound_interest,
    calculate_loan_payment,
    # ... and other finance tools
)

# 1. Create an agent with a full set of tools
finance_agent = create_agent(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    tools=[
        calculate_compound_interest,
        calculate_loan_payment,
        # ...
    ],
)

# 2. Initialize the Brain
brain = Brain(agent=finance_agent)`}
            </CodeBlock>
          </div>

          {/* Code Example 2: Generation */}
          <div className="mb-12">
            <h3 className="text-2xl font-bold text-[#7C3AED] mb-4">Step 2: Generation</h3>
            <p className="text-gray-400 mb-4">
              Next, we write a high-level description of the tasks we want to generate. We then call 
              <code className="bg-[#161B22] px-2 py-1 rounded text-[#7C3AED] mx-1">brain.generate_training_examples()</code> 
              and pass in our description and other parameters to control the output.
            </p>
            <CodeBlock
              language="python"
            >
{`# 3. Define the task description
task_description = (
    "Generate tasks to learn to use simple finance tools. "
    "The prompts should include varied numeric inputs and realistic edge cases."
)

# 4. Generate the training examples!
generated_examples = brain.generate_training_examples(
    task_description=task_description,
    num_examples=20,     # How many examples to create
    min_tool_calls=2,    # Ensure tasks are multi-step
    self_rank=True       # Use an LLM to rank the generated tasks for quality
)

# The output is a list of ready-to-use query strings
print(generated_examples[0])
# Expected output: "Calculate the monthly payment for a $250,000 loan over 30 years at a 6.5% annual rate, then find the compound interest on that payment over 5 years."`}
            </CodeBlock>
          </div>

          {/* Code Example 3: Training */}
          <div className="mb-12">
            <h3 className="text-2xl font-bold text-[#3FB950] mb-4">Step 3: Training</h3>
            <p className="text-gray-400 mb-4">
              Finally, the generated examples can be used directly to train your agent.
            </p>
            <CodeBlock
              language="python"
            >
{`# 5. Use the generated data for training
training_dataset = [{"query": example} for example in generated_examples]
brain.train(dataset=training_dataset)`}
            </CodeBlock>
          </div>
        </div>

        {/* Benefits Section */}
        <div className="mb-20">
          <h2 className="text-3xl font-bold text-[#E6EDF3] text-center mb-8">
            Why Zero-Learn Task Generation?
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">⚡</div>
              <h3 className="text-xl font-bold text-[#58A6FF] mb-3 text-center">Bootstrap Instantly</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Go from zero training data to a complete dataset in minutes, not hours or days of manual work.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-[#7C3AED]/10 to-[#6366F1]/10 border border-[#7C3AED]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">🎯</div>
              <h3 className="text-xl font-bold text-[#7C3AED] mb-3 text-center">Tool-Aligned Tasks</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Generated examples are specifically designed to exercise your agent's available tools effectively.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">🔄</div>
              <h3 className="text-xl font-bold text-[#3FB950] mb-3 text-center">Diverse & Realistic</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                LLM-generated tasks include edge cases and varied scenarios you might not think of manually.
              </p>
            </div>
          </div>
        </div>

        {/* CALL TO ACTION */}
        <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12 text-center">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            Ready to Generate Your Training Data?
          </h2>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Skip the tedious manual work and let ToolBrain create diverse, realistic training examples for your agent.
          </p>
          
          <a 
            href="/get-started/quickstart"
            className="inline-block bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-10 py-4 rounded-lg text-lg font-semibold transition-colors duration-200 shadow-lg hover:shadow-xl"
          >
            Get Started Now
          </a>
        </div>

      </div>
    </Layout>
  );
}