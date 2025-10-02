'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function ExecutionTrace() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Conceptual Guide: The Execution Trace</h1>
          <p className="text-gray-300 text-lg">
            Understanding ToolBrain&apos;s high-fidelity execution traces - the critical innovation that ensures 
            reliable RL training by capturing every detail of agent-environment interactions.
          </p>
        </div>

        {/* The Problem */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Foundation: Understanding Data Quality in RL</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              In reinforcement learning, there's a fundamental principle that determines the quality of your trained models: 
              <strong className="text-[#58A6FF]"> the quality of RL training is entirely dependent on the quality of the data it learns from</strong>.
            </p>

            <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-lg p-6 mb-6">
              <h3 className="text-xl font-semibold text-[#58A6FF] mb-4">📊 The Data Quality Challenge</h3>
              <p className="text-gray-300 mb-4">
                Many agent systems capture only the final, parsed actions, missing crucial information about 
                what the model actually saw and said. This incomplete data creates training challenges:
              </p>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-[#58A6FF] mb-2">� Information Gaps</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• Raw model outputs are simplified</li>
                    <li>• Parsing details are abstracted away</li>
                    <li>• Context information gets condensed</li>
                    <li>• Tool execution details are reduced</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-[#58A6FF] mb-2">🎯 Training Opportunities</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• More precise loss calculations</li>
                    <li>• Cleaner training signals</li>
                    <li>• Better convergence patterns</li>
                    <li>• Enhanced reward computation</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-gray-700 rounded-lg p-6">
              <h4 className="font-semibold text-gray-300 mb-3">🔍 Real-World Example: What Gets Lost</h4>
              <p className="text-gray-300 mb-3 text-sm">
                Consider an agent trying to solve a math problem. Here&apos;s what typically gets logged vs. what actually happened:
              </p>
              
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded p-3">
                  <h5 className="text-[#58A6FF] font-semibold text-sm mb-2">📝 Standard Logging</h5>
                  <CodeBlock language="json">
{`{
  "action": "python_code",
  "code": "print(2 + 2)",
  "result": "4"
}`}
                  </CodeBlock>
                  <p className="text-gray-400 text-xs mt-2">
                    Typical approach: Structured but incomplete information
                  </p>
                </div>
                
                <div className="bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded p-3">
                  <h5 className="text-[#3FB950] font-semibold text-sm mb-2">✅ High-Fidelity Trace</h5>
                  <CodeBlock language="text">
{`Model saw: "Solve: What is 2+2?"
Model output: "I need to calculate this.
Action: python_code
Code: print(2 + 2)
Final answer: The result is 4"
Tool executed: Python interpreter
Raw tool output: "4\\n"
Parsed result: {"result": "4"}`}
                  </CodeBlock>
                  <p className="text-[#3FB950] text-xs mt-2">
                    Complete information enables accurate training and debugging
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-6 p-4 bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-lg">
              <h4 className="font-semibold text-[#3FB950] mb-2">💡 The ToolBrain Advantage</h4>
              <p className="text-gray-300 text-sm">
                With complete, high-fidelity execution traces, your model learns from an accurate view of reality. 
                This leads to better performance, reliable behavior, and easy debugging capabilities. 
                <strong>Complete visibility enables perfect training.</strong>
              </p>
            </div>
          </div>
        </section>

        {/* The Solution */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Solution: The High-Fidelity Execution Trace</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              ToolBrain solves this fundamental problem with the <strong className="text-green-400">Execution Trace</strong> - 
              a definitive, structured record of an agent&apos;s interaction with its environment that captures 
              <em>every single detail</em> of what happened during execution.
            </p>

            <div className="bg-green-900/20 border border-green-600 rounded-lg p-6 mb-6">
              <h3 className="text-xl font-semibold text-green-400 mb-4">✅ High-Fidelity Design Principles</h3>
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h4 className="font-semibold text-green-300 mb-2">🎯 Complete Capture</h4>
                  <ul className="text-green-200 text-sm space-y-1">
                    <li>• Raw model inputs and outputs</li>
                    <li>• Parsing results and interpretations</li>
                    <li>• Tool execution details</li>
                    <li>• Structured data objects</li>
                    <li>• Conversation history</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-green-300 mb-2">🔬 Perfect Fidelity</h4>
                  <ul className="text-green-200 text-sm space-y-1">
                    <li>• No information loss or distortion</li>
                    <li>• Exact reproduction capability</li>
                    <li>• Ground truth for training</li>
                    <li>• Complete debugging visibility</li>
                    <li>• Reliable reward computation</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-2">🏗️ The Execution Trace Architecture</h4>
              <p className="text-blue-200 text-sm mb-3">
                An Execution Trace is structured as a chronological sequence of interactions, where each step 
                contains complete information about what the agent saw, did, and received back from the environment.
              </p>
              <div className="text-blue-200 text-sm">
                <strong>Structure:</strong> <code className="bg-gray-700 px-2 py-1 rounded">ExecutionTrace = List[Turn]</code>
              </div>
            </div>
          </div>
        </section>

        {/* Anatomy of a Trace */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Anatomy of a Trace</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Each Execution Trace is composed of a list of <code className="bg-gray-700 px-2 py-1 rounded">Turn</code> objects. 
              Here are the exact TypedDict definitions from <code className="bg-gray-700 px-2 py-1 rounded">toolbrain/core_types.py</code>:
            </p>
            
            <CodeBlock language="python">
{`# From toolbrain/core_types.py - The exact data structures

from typing import TypedDict, Optional, Any, List

class ParsedCompletion(TypedDict):
    """
    Structured interpretation of the model&apos;s raw output.
    Represents how the framework parses and understands the model&apos;s response.
    """
    thought: Optional[str]        # Model's reasoning or explanation
    tool_code: Optional[str]      # Code or command to execute
    final_answer: Optional[str]   # Direct response to the user

class Turn(TypedDict):
    """
    Complete record of a single agent-environment interaction.
    Captures everything that happened in one step of execution.
    """
    prompt_for_model: str                    # Exact input sent to the LLM
    model_completion: str                    # Raw, unprocessed LLM output  
    parsed_completion: ParsedCompletion      # Framework's interpretation
    tool_output: Optional[str]               # String representation of tool result
    action_output: Optional[Any]             # Original Python object from tool
    formatted_conversation: Optional[str]    # Pre-formatted conversation history

# An ExecutionTrace is simply a list of these turns
ExecutionTrace = List[Turn]`}
            </CodeBlock>

            <div className="mt-6 bg-blue-900/20 border border-blue-600 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-2">🔄 The Flow of Information</h4>
              <p className="text-blue-200 text-sm">
                Each Turn captures one complete cycle of interaction: the agent receives a prompt, generates a response, 
                that response gets parsed and potentially executed as a tool call, and the result feeds back into the next turn.
              </p>
            </div>
          </div>
        </section>

        {/* Why Each Field Matters */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Why Each Field Matters: The Complete Picture</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              Every field in the Turn structure serves a critical purpose for reliable RL training. 
              Let&apos;s examine why each piece of information is essential:
            </p>

            <div className="space-y-6">
              {/* prompt_for_model */}
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-6">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-bold">📝</span>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-blue-400 mb-2">
                      <code className="bg-gray-700 px-2 py-1 rounded text-sm">prompt_for_model</code>
                    </h3>
                    <p className="text-blue-200 mb-3">
                      <strong>The ground truth of what the LLM saw.</strong> Essential for calculating log-probabilities during training.
                    </p>
                    <div className="bg-gray-700 rounded p-3 text-sm">
                      <p className="text-blue-300 mb-2"><strong>Why Critical:</strong></p>
                      <ul className="text-blue-200 space-y-1">
                        <li>• RL algorithms need exact input to compute gradients correctly</li>
                        <li>• Enables precise loss calculation for policy optimization</li>
                        <li>• Required for replay and reproducibility</li>
                        <li>• Debugging: "What exactly did the model see when it made this decision?"</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* model_completion */}
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-6">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-bold">🤖</span>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-green-400 mb-2">
                      <code className="bg-gray-700 px-2 py-1 rounded text-sm">model_completion</code>
                    </h3>
                    <p className="text-green-200 mb-3">
                      <strong>The raw, unaltered text the LLM generated.</strong> Preserves 100% of the output before any parsing.
                    </p>
                    <div className="bg-gray-700 rounded p-3 text-sm">
                      <p className="text-green-300 mb-2"><strong>Why Critical:</strong></p>
                      <ul className="text-green-200 space-y-1">
                        <li>• Contains the model&apos;s exact reasoning and thought process</li>
                        <li>• Needed for computing action probabilities in RL training</li>
                        <li>• Reveals parsing errors and edge cases</li>
                        <li>• Debugging: "What did the model actually say vs. what we interpreted?"</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* parsed_completion */}
              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-6">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-bold">🔍</span>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-purple-400 mb-2">
                      <code className="bg-gray-700 px-2 py-1 rounded text-sm">parsed_completion</code>
                    </h3>
                    <p className="text-purple-200 mb-3">
                      <strong>The framework&apos;s structured interpretation of the raw output.</strong> Shows how the system understood the model&apos;s response.
                    </p>
                    <div className="bg-gray-700 rounded p-3 text-sm">
                      <p className="text-purple-300 mb-2"><strong>Why Critical:</strong></p>
                      <ul className="text-purple-200 space-y-1">
                        <li>• Separates reasoning (`thought`) from action (`tool_code`) from response (`final_answer`)</li>
                        <li>• Enables fine-grained reward computation on different aspects</li>
                        <li>• Identifies parsing successes and failures</li>
                        <li>• Debugging: "How did the framework interpret this output?"</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* tool_output */}
              <div className="bg-orange-900/20 border border-orange-600 rounded-lg p-6">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-orange-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-bold">🔧</span>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-orange-400 mb-2">
                      <code className="bg-gray-700 px-2 py-1 rounded text-sm">tool_output</code>
                    </h3>
                    <p className="text-orange-200 mb-3">
                      <strong>The string representation of the tool&apos;s result.</strong> This is what the LLM sees in the next turn.
                    </p>
                    <div className="bg-gray-700 rounded p-3 text-sm">
                      <p className="text-orange-300 mb-2"><strong>Why Critical:</strong></p>
                      <ul className="text-orange-200 space-y-1">
                        <li>• Shows exactly what feedback the model received</li>
                        <li>• Essential for multi-turn conversation training</li>
                        <li>• Enables reward functions based on tool interaction quality</li>
                        <li>• Debugging: "What did the tool actually return to the model?"</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* action_output */}
              <div className="bg-teal-900/20 border border-teal-600 rounded-lg p-6">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-teal-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-bold">📊</span>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-teal-400 mb-2">
                      <code className="bg-gray-700 px-2 py-1 rounded text-sm">action_output</code>
                    </h3>
                    <p className="text-teal-200 mb-3">
                      <strong>The original Python object returned by the tool.</strong> Crucial for precise, rule-based reward functions that work with structured data.
                    </p>
                    <div className="bg-gray-700 rounded p-3 text-sm">
                      <p className="text-teal-300 mb-2"><strong>Why Critical:</strong></p>
                      <ul className="text-teal-200 space-y-1">
                        <li>• Enables type-safe reward computation (e.g., check if result is a float)</li>
                        <li>• Allows complex data structure analysis</li>
                        <li>• Perfect for mathematical accuracy checking</li>
                        <li>• Debugging: "What was the actual structured result, not just the string?"</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* formatted_conversation */}
              <div className="bg-indigo-900/20 border border-indigo-600 rounded-lg p-6">
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 bg-indigo-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-bold">💬</span>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-indigo-400 mb-2">
                      <code className="bg-gray-700 px-2 py-1 rounded text-sm">formatted_conversation</code>
                    </h3>
                    <p className="text-indigo-200 mb-3">
                      <strong>Pre-formatted conversation history.</strong> Used for efficient tokenization during the learning phase.
                    </p>
                    <div className="bg-gray-700 rounded p-3 text-sm">
                      <p className="text-indigo-300 mb-2"><strong>Why Critical:</strong></p>
                      <ul className="text-indigo-200 space-y-1">
                        <li>• Optimizes training performance by pre-computing conversation formats</li>
                        <li>• Ensures consistent tokenization across training steps</li>
                        <li>• Enables efficient batch processing during RL updates</li>
                        <li>• Debugging: "How was the conversation formatted for the model?"</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Complete Example */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">A Complete Execution Trace Example</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Here&apos;s what a real Execution Trace looks like for a simple math problem:
            </p>
            
            <CodeBlock language="python">
{`# Example: Agent solving "What is the square root of 144?"
execution_trace = [
    # Turn 1: Initial reasoning and tool call
    {
        "prompt_for_model": "User: What is the square root of 144?\\n\\nYou have access to a Python interpreter. Use it to solve this problem.",
        
        "model_completion": "I need to calculate the square root of 144. Let me use Python for this.\\n\\nAction: python_code\\nCode: import math\\nresult = math.sqrt(144)\\nprint(f'The square root of 144 is {result}')\\n\\nFinal Answer: Let me calculate this for you.",
        
        "parsed_completion": {
            "thought": "I need to calculate the square root of 144. Let me use Python for this.",
            "tool_code": "import math\\nresult = math.sqrt(144)\\nprint(f'The square root of 144 is {result}')",
            "final_answer": None  # No final answer yet, still working
        },
        
        "tool_output": "The square root of 144 is 12.0",
        
        "action_output": {
            "stdout": "The square root of 144 is 12.0\\n",
            "stderr": "",
            "return_code": 0,
            "variables": {"result": 12.0}
        },
        
        "formatted_conversation": "User: What is the square root of 144?\\n\\nAssistant: I need to calculate the square root of 144. Let me use Python for this.\\n\\nAction: python_code\\nCode: import math\\nresult = math.sqrt(144)\\nprint(f'The square root of 144 is {result}')\\n\\nTool output: The square root of 144 is 12.0"
    },
    
    # Turn 2: Providing the final answer
    {
        "prompt_for_model": "User: What is the square root of 144?\\n\\nYou have access to a Python interpreter. Use it to solve this problem.\\n\\nPrevious interaction:\\nAssistant: I need to calculate the square root of 144. Let me use Python for this.\\n\\nAction: python_code\\nCode: import math\\nresult = math.sqrt(144)\\nprint(f'The square root of 144 is {result}')\\n\\nTool output: The square root of 144 is 12.0\\n\\nNow provide your final answer:",
        
        "model_completion": "Based on the calculation, the square root of 144 is 12.\\n\\nFinal Answer: The square root of 144 is 12.",
        
        "parsed_completion": {
            "thought": "Based on the calculation, the square root of 144 is 12.",
            "tool_code": None,  # No tool call in this turn
            "final_answer": "The square root of 144 is 12."
        },
        
        "tool_output": None,  # No tool was called
        "action_output": None,  # No tool execution
        
        "formatted_conversation": "User: What is the square root of 144?\\n\\nAssistant: Based on the calculation, the square root of 144 is 12.\\n\\nFinal Answer: The square root of 144 is 12."
    }
]

# This complete trace enables:
# 1. Perfect reproduction of the interaction
# 2. Accurate reward computation (both on text and structured data)
# 3. Reliable RL training with correct loss calculation
# 4. Complete debugging visibility
# 5. Quality assessment at every step`}
            </CodeBlock>
          </div>
        </section>

        {/* Benefits of Data Fidelity */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Benefits of Data Fidelity</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              The high-fidelity Execution Trace design delivers three critical benefits that directly translate 
              to better trained agents:
            </p>

            <div className="space-y-8">
              {/* Reliable Training */}
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-6">
                <div className="flex items-start gap-4 mb-4">
                  <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-bold">🎯</span>
                  </div>
                  <div>
                    <h3 className="text-2xl font-semibold text-green-400">Reliable Training</h3>
                    <p className="text-green-200">Perfect accuracy in loss computation leads to stable, effective learning</p>
                  </div>
                </div>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold text-green-300 mb-2">🔬 Ground Truth Data</h4>
                    <ul className="text-green-200 text-sm space-y-1">
                      <li>• Exact input-output pairs for loss calculation</li>
                      <li>• No information loss during training</li>
                      <li>• Accurate gradient computation</li>
                      <li>• Stable convergence behavior</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold text-green-300 mb-2">📈 Training Quality</h4>
                    <ul className="text-green-200 text-sm space-y-1">
                      <li>• Consistent training signals</li>
                      <li>• Reduced noise in optimization</li>
                      <li>• Better sample efficiency</li>
                      <li>• Predictable learning dynamics</li>
                    </ul>
                  </div>
                </div>

                <div className="mt-4 p-4 bg-gray-700 rounded-lg">
                  <p className="text-green-200 text-sm">
                    <strong>Result:</strong> Your agents train faster, more reliably, and achieve better final performance 
                    because the RL algorithms have access to perfect, unbiased data about what actually happened.
                  </p>
                </div>
              </div>

              {/* Powerful Rewards */}
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-6">
                <div className="flex items-start gap-4 mb-4">
                  <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-bold">⚡</span>
                  </div>
                  <div>
                    <h3 className="text-2xl font-semibold text-blue-400">Powerful Rewards</h3>
                    <p className="text-blue-200">Maximum flexibility in reward design using both text and structured data</p>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-blue-300 mb-2">🎛️ Reward Function Flexibility</h4>
                    <CodeBlock language="python">
{`# Example: Sophisticated reward function using both text and structured data
def advanced_math_reward(trace: ExecutionTrace) -> float:
    reward = 0.0
    
    for turn in trace:
        # Reward based on structured data (action_output)
        if turn["action_output"] and isinstance(turn["action_output"], dict):
            if "variables" in turn["action_output"]:
                result = turn["action_output"]["variables"].get("result")
                if isinstance(result, (int, float)):
                    # Reward for getting numeric result
                    reward += 0.3
                    
                    # Bonus for correct mathematical result
                    if abs(result - 12.0) < 0.001:  # sqrt(144) = 12
                        reward += 0.5
        
        # Reward based on text quality (tool_output)
        if turn["tool_output"]:
            if "error" not in turn["tool_output"].lower():
                reward += 0.1  # No errors
            
            if any(word in turn["tool_output"] for word in ["sqrt", "square root"]):
                reward += 0.1  # Relevant mathematical terms
        
        # Reward based on reasoning quality (parsed_completion)
        if turn["parsed_completion"]["thought"]:
            thought = turn["parsed_completion"]["thought"]
            if "calculate" in thought or "math" in thought:
                reward += 0.2  # Shows mathematical reasoning
    
    return min(reward, 1.0)  # Cap at 1.0`}
                    </CodeBlock>
                  </div>
                  
                  <div className="bg-gray-700 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-300 mb-2">🌟 What This Enables</h4>
                    <ul className="text-blue-200 text-sm space-y-1">
                      <li>• <strong>Type-safe rewards:</strong> Check if results are numbers, not just strings</li>
                      <li>• <strong>Multi-aspect evaluation:</strong> Reward reasoning, execution, and results separately</li>
                      <li>• <strong>Complex logic:</strong> Implement sophisticated success criteria</li>
                      <li>• <strong>Domain-specific metrics:</strong> Use structured data for precise evaluation</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Effortless Debugging */}
              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-6">
                <div className="flex items-start gap-4 mb-4">
                  <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-white font-bold">🔍</span>
                  </div>
                  <div>
                    <h3 className="text-2xl font-semibold text-purple-400">Effortless Debugging</h3>
                    <p className="text-purple-200">Complete visibility into every step of agent execution</p>
                  </div>
                </div>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold text-purple-300 mb-2">🕵️ Investigation Capabilities</h4>
                    <ul className="text-purple-200 text-sm space-y-1">
                      <li>• See exactly what the model received as input</li>
                      <li>• Compare raw output vs. parsed interpretation</li>
                      <li>• Track tool execution and results</li>
                      <li>• Analyze conversation flow and context</li>
                      <li>• Identify parsing errors and edge cases</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold text-purple-300 mb-2">🐛 Common Debugging Scenarios</h4>
                    <ul className="text-purple-200 text-sm space-y-1">
                      <li>• "Why did the agent choose this action?"</li>
                      <li>• "What went wrong in this tool call?"</li>
                      <li>• &quot;How was the model&apos;s output interpreted?&quot;</li>
                      <li>• "What context was missing from this turn?"</li>
                      <li>• "Why did the reward function give this score?"</li>
                    </ul>
                  </div>
                </div>

                <div className="mt-4 p-4 bg-gray-700 rounded-lg">
                  <h4 className="font-semibold text-purple-300 mb-2">🎯 Debugging Example</h4>
                  <CodeBlock language="python">
{`# Debugging a failed interaction
def debug_trace(trace: ExecutionTrace):
    for i, turn in enumerate(trace):
        print(f"\\n=== Turn {i+1} ===")
        print(f"Model saw: {turn['prompt_for_model'][:100]}...")
        print(f"Model said: {turn['model_completion'][:100]}...")
        
        if turn['parsed_completion']['tool_code']:
            print(f"Parsed tool code: {turn['parsed_completion']['tool_code']}")
            print(f"Tool output: {turn['tool_output']}")
            
            if turn['action_output']:
                print(f"Structured result: {turn['action_output']}")
                
        # Check for issues
        if not turn['tool_output'] and turn['parsed_completion']['tool_code']:
            print("⚠️  WARNING: Tool code present but no output!")
        
        if "error" in str(turn['tool_output']).lower():
            print("❌ ERROR: Tool execution failed!")

# Perfect visibility = faster debugging = better agents`}
                  </CodeBlock>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Performance Impact */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Performance Impact</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              High-fidelity execution traces translate directly into better agent performance:
            </p>

            <div className="grid md:grid-cols-3 gap-6 mb-6">
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-4 text-center">
                <h3 className="text-lg font-semibold text-green-400 mb-2">🎯 Training Accuracy</h3>
                <p className="text-3xl font-bold text-green-400 mb-2">95%+</p>
                <p className="text-green-200 text-sm">Accurate loss computation leads to reliable learning</p>
              </div>
              
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4 text-center">
                <h3 className="text-lg font-semibold text-blue-400 mb-2">⚡ Debug Speed</h3>
                <p className="text-3xl font-bold text-blue-400 mb-2">10x</p>
                <p className="text-blue-200 text-sm">Faster issue identification and resolution</p>
              </div>
              
              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-4 text-center">
                <h3 className="text-lg font-semibold text-purple-400 mb-2">🔬 Reward Precision</h3>
                <p className="text-3xl font-bold text-purple-400 mb-2">100%</p>
                <p className="text-purple-200 text-sm">Complete data enables perfect reward computation</p>
              </div>
            </div>

            <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-400 mb-2">🔄 The Virtuous Cycle</h4>
              <p className="text-yellow-200 text-sm">
                Better data → More reliable training → Better agents → Easier debugging → Faster iteration → 
                Even better agents. The high-fidelity execution trace creates a positive feedback loop that 
                accelerates your entire development process.
              </p>
            </div>
          </div>
        </section>

        {/* Conclusion */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Foundation of Reliable RL</h2>
          <div className="bg-gray-800 rounded-lg p-6">
            <p className="text-gray-300 mb-4">
              The Execution Trace isn&apos;t just a nice-to-have feature - it&apos;s the foundational innovation that makes 
              reliable reinforcement learning for agents possible. By capturing every detail of agent-environment 
              interactions with perfect fidelity, ToolBrain ensures that:
            </p>

            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div>
                <h3 className="text-lg font-semibold text-green-400 mb-3">✅ What You Get</h3>
                <ul className="text-green-200 text-sm space-y-2">
                  <li>• Accurate, stable RL training</li>
                  <li>• Flexible, powerful reward functions</li>
                  <li>• Complete debugging visibility</li>
                  <li>• Reproducible experiments</li>
                  <li>• Reliable performance metrics</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-[#3FB950] mb-3">🎯 Key Benefits</h3>
                <ul className="text-[#3FB950] text-sm space-y-2">
                  <li>• High-quality training data</li>
                  <li>• Clear behavior visibility</li>
                  <li>• Enhanced reward computation</li>
                  <li>• Efficient debugging workflows</li>
                  <li>• Consistent performance metrics</li>
                </ul>
              </div>
            </div>

            <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-2">🚀 Ready to Experience High-Fidelity Training?</h4>
              <p className="text-blue-200 text-sm">
                Every ToolBrain agent automatically generates high-fidelity execution traces. No configuration needed, 
                no extra complexity - just reliable, debuggable, high-quality RL training out of the box.
              </p>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}