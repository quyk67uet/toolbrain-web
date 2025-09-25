'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function UnifiedAPI() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Guide: The Unified API & 'Coach-Athlete' Architecture</h1>
          <p className="text-gray-300 text-lg">
            Understanding ToolBrain's core design philosophy and unified API that makes training intelligent agents simple and consistent.
          </p>
        </div>

        {/* Architecture Overview */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The 'Coach-Athlete' Architecture</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              ToolBrain implements a 'Coach-Athlete' paradigm where the <strong>Brain</strong> acts as an intelligent coach 
              that trains and optimizes <strong>Agent</strong> athletes. This separation of concerns creates a clean, 
              extensible architecture.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-blue-400 mb-3">🧠 The Brain (Coach)</h3>
                <ul className="text-blue-200 text-sm space-y-2">
                  <li>• Central orchestrator and trainer</li>
                  <li>• Manages training loops and optimization</li>
                  <li>• Handles reward engineering</li>
                  <li>• Coordinates tool retrieval</li>
                  <li>• Generates training examples</li>
                </ul>
              </div>
              
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-green-400 mb-3">🤖 The Agent (Athlete)</h3>
                <ul className="text-green-200 text-sm space-y-2">
                  <li>• Executes tasks and learns</li>
                  <li>• Interacts with tools and environment</li>
                  <li>• Generates responses and actions</li>
                  <li>• Adapts based on feedback</li>
                  <li>• Focuses on task execution</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        {/* The Brain API */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Brain API: Central Orchestrator</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              The <code className="bg-gray-700 px-2 py-1 rounded">Brain</code> class serves as the central orchestrator, 
              providing a unified interface for all training operations. Here's the core <code className="bg-gray-700 px-2 py-1 rounded">__init__</code> method:
            </p>
            
            <CodeBlock language="python">
{`class Brain:
    def __init__(
        self,
        agent,
        reward_func=None,
        enable_tool_retrieval=False,
        learning_algorithm="GRPO",
        batch_size=32,
        learning_rate=1e-4,
        **kwargs
    ):
        """
        Initialize the Brain for agent training.
        
        Args:
            agent: The agent to train (CodeAgent, ReActAgent, etc.)
            reward_func: Function to compute rewards for agent actions
            enable_tool_retrieval: Whether to enable intelligent tool retrieval
            learning_algorithm: Training algorithm ("GRPO", "DPO", "SFT")
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
        """
        self.agent = agent
        self.reward_func = reward_func
        self.enable_tool_retrieval = enable_tool_retrieval
        self.learning_algorithm = learning_algorithm
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        `}
            </CodeBlock>

            <div className="mt-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Key Parameters Explained</h3>
              <div className="space-y-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-green-400 mb-2">agent</h4>
                  <p className="text-gray-300 text-sm">
                    The agent instance to be trained. Can be any agent type (CodeAgent, ReActAgent, etc.). 
                    The Brain automatically adapts to different agent architectures.
                  </p>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-green-400 mb-2">reward_func</h4>
                  <p className="text-gray-300 text-sm">
                    Function to evaluate agent performance. Can be user-defined functions or LLM-as-a-judge. 
                    If None, uses environment rewards.
                  </p>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-green-400 mb-2">enable_tool_retrieval</h4>
                  <p className="text-gray-300 text-sm">
                    When True, automatically retrieves relevant tools for each task, solving the "too many tools" problem.
                  </p>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-green-400 mb-2">learning_algorithm</h4>
                  <p className="text-gray-300 text-sm">
                    Training algorithm to use. Supports "GRPO" (Generalized RPO), "DPO" (Direct Preference Optimization), 
                    and "SFT" (Supervised Fine-Tuning).
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* The Agent Adapter Pattern */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Agent Adapter Pattern</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              ToolBrain uses the Adapter Pattern to seamlessly work with different agent types. The Brain automatically 
              detects the agent type and applies the appropriate adapter for training.
            </p>
            
            <CodeBlock language="python">
{`def _get_adapter_for_agent(self, agent):
    """
    Automatically select the appropriate adapter for the given agent.
    
    Args:
        agent: The agent instance
        
    Returns:
        Adapter: The appropriate adapter for this agent type
    """
    if isinstance(agent, CodeAgent):
        return CodeAgentAdapter(agent)
    elif isinstance(agent, ReActAgent):
        return ReActAgentAdapter(agent)
    elif isinstance(agent, ConversationalAgent):
        return ConversationalAgentAdapter(agent)
    elif hasattr(agent, 'generate'):
        # Generic LLM-based agent
        return GenericLLMAdapter(agent)
    else:
        raise ValueError(f"Unsupported agent type: {type(agent)}")
        
# Usage in Brain initialization
def __init__(self, agent, **kwargs):
    self.agent = agent
    self.adapter = self._get_adapter_for_agent(agent)
    # The adapter handles agent-specific training logic`}
            </CodeBlock>

            <div className="mt-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Supported Agent Types</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-purple-400 mb-2">CodeAgent</h4>
                  <p className="text-gray-300 text-sm">
                    Specialized for code generation tasks. Handles tool usage, code execution, and debugging workflows.
                  </p>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-purple-400 mb-2">ReActAgent</h4>
                  <p className="text-gray-300 text-sm">
                    Implements Reasoning + Acting paradigm. Alternates between reasoning and action steps.
                  </p>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-purple-400 mb-2">ConversationalAgent</h4>
                  <p className="text-gray-300 text-sm">
                    Optimized for dialogue and conversation tasks. Handles context and multi-turn interactions.
                  </p>
                </div>
                
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-purple-400 mb-2">Generic LLM</h4>
                  <p className="text-gray-300 text-sm">
                    Fallback adapter for any agent with a 'generate' method. Provides basic training capabilities.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Unified Workflow */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Unified Training Workflow</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              Regardless of the agent type, ToolBrain provides a consistent training workflow through the unified API:
            </p>
            
            <CodeBlock language="python">
{`# 1. Initialize any agent type
code_agent = CodeAgent(model="Qwen/3B-Instruct", tools=finance_tools)
react_agent = ReActAgent(model="Llama-3-8B", tools=search_tools)
chat_agent = ConversationalAgent(model="Mistral-7B")

# 2. Create Brain with unified API - same for all agent types
brain_code = Brain(
    agent=code_agent,
    reward_func=finance_reward,
    enable_tool_retrieval=True,
    learning_algorithm="GRPO"
)

brain_react = Brain(
    agent=react_agent,
    reward_func=search_reward,
    enable_tool_retrieval=True,
    learning_algorithm="DPO"
)

brain_chat = Brain(
    agent=chat_agent,
    reward_func=conversation_reward,
    learning_algorithm="SFT"
)

# 3. Same training interface for all
for brain in [brain_code, brain_react, brain_chat]:
    # Generate training data
    training_data = brain.generate_training_examples(
        task_description=f"Master the {brain.agent.__class__.__name__} tasks"
    )
    
    # Optional: Knowledge distillation
    brain.distill(dataset=training_data, teacher_model_id="GPT-4-Turbo")
    
    # Train the agent
    brain.train(dataset=training_data)
    
    # Evaluate performance
    results = brain.evaluate(test_dataset)`}
            </CodeBlock>

            <div className="mt-6 p-4 bg-green-900/20 border border-green-600 rounded-lg">
              <h4 className="font-semibold text-green-400 mb-2">🎯 Key Benefits</h4>
              <ul className="text-green-200 text-sm space-y-1">
                <li>• <strong>Consistency:</strong> Same API works across all agent types</li>
                <li>• <strong>Flexibility:</strong> Easy to switch between different agents and algorithms</li>
                <li>• <strong>Extensibility:</strong> Simple to add new agent types via adapters</li>
                <li>• <strong>Maintainability:</strong> Centralized training logic in the Brain</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Real-World Example */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Real-World Example</h2>
          <div className="bg-gray-800 rounded-lg p-6">
            <p className="text-gray-300 mb-4">
              Here's a complete example showing how the unified API handles different agent types seamlessly:
            </p>
            
            <CodeBlock language="python">
{`from toolbrain import Brain
from toolbrain.agents import CodeAgent, ReActAgent
from toolbrain.rewards import reward_llm_judge_via_ranking

# Define tools and rewards
finance_tools = [get_stock_price, calculate_portfolio, risk_analysis]
research_tools = [web_search, academic_search, summarize_paper]

# Create different agent types
code_agent = CodeAgent(
    model="Qwen/3B-Instruct",
    tools=finance_tools
)

research_agent = ReActAgent(
    model="Llama-3-8B", 
    tools=research_tools
)

# Use the same Brain API for both
brains = []
for agent, task_desc in [
    (code_agent, "Master financial analysis and portfolio management"),
    (research_agent, "Conduct comprehensive market research")
]:
    brain = Brain(
        agent=agent,
        reward_func=reward_llm_judge_via_ranking,
        enable_tool_retrieval=True,
        learning_algorithm="GRPO",
        max_steps=1000
    )
    
    # Same workflow regardless of agent type
    training_data = brain.generate_training_examples(task_description=task_desc)
    brain.distill(dataset=training_data, teacher_model_id="GPT-4-Turbo")
    brain.train(dataset=training_data)
    
    brains.append(brain)

print("Both agents trained successfully with the same unified API!")`}
            </CodeBlock>

            <div className="mt-6 p-4 bg-blue-900/20 border border-blue-600 rounded-lg">
              <h4 className="font-semibold text-blue-400 mb-2">🚀 The Power of Unified API</h4>
              <p className="text-blue-200 text-sm">
                This example demonstrates how ToolBrain's unified API allows you to train completely different types of agents 
                (CodeAgent vs ReActAgent) using identical code patterns. The Brain handles all the complexity of adapting 
                to different agent architectures behind the scenes.
              </p>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}