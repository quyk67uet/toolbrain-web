'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import Image from 'next/image';

export default function CoachAthleteParadigm() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Conceptual Guide: The 'Coach-Athlete' Paradigm</h1>
          <p className="text-gray-300 text-lg">
            Understanding ToolBrain's core architectural philosophy through the lens of sports coaching - 
            where separation of concerns enables both simplicity and extensibility.
          </p>
        </div>

        {/* System Diagram */}
        <section className="mb-12">
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <div className="flex justify-center mb-4">
              <Image
                src="/system_toolbrain_w.jpg"
                alt="ToolBrain System Architecture - Coach-Athlete Paradigm"
                width={800}
                height={600}
                className="rounded-lg border border-gray-600"
              />
            </div>
            <p className="text-gray-300 text-center text-sm">
              <strong>The ToolBrain Architecture:</strong> A clear separation between The Coach (Brain), The Athlete (Agent), 
              and The Interpreter (Adapter), enabling powerful yet simple reinforcement learning for agents.
            </p>
          </div>
        </section>

        {/* The Problem */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Problem: The Tangle of Agent & RL Logic</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              In many existing systems, the agent's task-solving logic is tightly coupled with the reinforcement learning logic. 
              This creates several critical challenges that hinder both usability and extensibility:
            </p>

            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-red-900/20 border border-red-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-red-400 mb-3">❌ Tight Coupling Problems</h3>
                <ul className="text-red-200 text-sm space-y-2">
                  <li>• <strong>Hard to maintain:</strong> Changes to agent logic require RL expertise</li>
                  <li>• <strong>Difficult to extend:</strong> Adding new agent types means rewriting RL components</li>
                  <li>• <strong>High barrier to entry:</strong> Developers need deep RL knowledge for basic tasks</li>
                  <li>• <strong>Code duplication:</strong> Similar RL logic scattered across different agent types</li>
                  <li>• <strong>Testing complexity:</strong> Can't test agent logic independently from training</li>
                  <li>• <strong>Poor modularity:</strong> Everything depends on everything else</li>
                </ul>
              </div>
              
              <div className="bg-gray-700 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-gray-300 mb-3">🔍 Real-World Example</h3>
                <p className="text-gray-300 text-sm mb-3">
                  Consider a typical tightly-coupled system:
                </p>
                <CodeBlock language="python">
{`# Typical tightly-coupled approach ❌
class CodeAgentWithRL:
    def __init__(self):
        self.model = load_model()
        self.tools = load_tools()
        # RL components mixed in
        self.optimizer = Adam()
        self.replay_buffer = Buffer()
        self.value_network = ValueNet()
    
    def solve_task(self, query):
        # Task logic mixed with RL logic
        action = self.model.generate(query)
        reward = self.compute_reward(action)
        self.update_policy(reward)  # RL update
        return self.execute_tools(action)
    
    def compute_reward(self, action):
        # Reward logic tightly coupled
        pass
    
    def update_policy(self, reward):
        # RL update logic embedded
        pass`}
                </CodeBlock>
                <p className="text-gray-400 text-xs mt-2">
                  This approach makes it impossible to reuse the RL training logic with different agent frameworks.
                </p>
              </div>
            </div>

            <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-4">
              <h4 className="font-semibold text-yellow-400 mb-2">🚧 The Consequences</h4>
              <p className="text-yellow-200 text-sm">
                When agent logic and RL logic are intertwined, every new agent framework requires a complete rewrite 
                of the training infrastructure. This leads to fragmented ecosystems where each agent type has its own 
                training approach, making it nearly impossible to leverage advances across different frameworks.
              </p>
            </div>
          </div>
        </section>

        {/* The Solution */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Solution: A Separation of Concerns</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              ToolBrain introduces the <strong className="text-blue-400">'Coach-Athlete' paradigm</strong> as an elegant 
              solution to this architectural challenge. This metaphor provides a clear mental model for understanding 
              how different responsibilities are separated:
            </p>

            <div className="space-y-8">
              {/* The Athlete */}
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-green-600 rounded-full flex items-center justify-center mr-4">
                    <span className="text-2xl">🏃‍♂️</span>
                  </div>
                  <h3 className="text-2xl font-semibold text-green-400">The Athlete (Agent)</h3>
                </div>
                <p className="text-green-200 mb-4">
                  <strong>Single Responsibility:</strong> Perform the task with expertise in its domain
                </p>
                <ul className="text-green-200 text-sm space-y-2 mb-4">
                  <li>• <strong>Domain expert:</strong> Knows how to call tools, solve problems, interact with APIs</li>
                  <li>• <strong>Task-focused:</strong> Only cares about executing the current task well</li>
                  <li>• <strong>Unaware of training:</strong> Has no knowledge that it's being trained or evaluated</li>
                  <li>• <strong>Framework agnostic:</strong> Can be any agent from any framework (SmolAgents, LangGraph, etc.)</li>
                  <li>• <strong>Stateless regarding RL:</strong> Doesn't maintain any training-related state</li>
                </ul>
                
                <CodeBlock language="python">
{`# The Athlete: Pure task execution ✅
from smolagents import CodeAgent

# This is a pure agent - no RL logic mixed in
athlete = CodeAgent(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    tools=[
        "python_interpreter",
        "web_search", 
        "file_reader"
    ]
)

# The agent just does what it does best - solve tasks
result = athlete.run("Calculate the fibonacci sequence up to n=20")
# No RL updates, no reward computation, just pure task execution`}
                </CodeBlock>
              </div>

              {/* The Coach */}
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-blue-600 rounded-full flex items-center justify-center mr-4">
                    <span className="text-2xl">🧠</span>
                  </div>
                  <h3 className="text-2xl font-semibold text-blue-400">The Coach (Brain)</h3>
                </div>
                <p className="text-blue-200 mb-4">
                  <strong>Single Responsibility:</strong> Observe, evaluate, and improve the Athlete's performance
                </p>
                <ul className="text-blue-200 text-sm space-y-2 mb-4">
                  <li>• <strong>Training expert:</strong> Understands RL algorithms, optimization, and learning strategies</li>
                  <li>• <strong>Performance monitor:</strong> Observes the Athlete's actions and outcomes</li>
                  <li>• <strong>Feedback provider:</strong> Computes rewards and provides training signals</li>
                  <li>• <strong>Strategy developer:</strong> Uses a "playbook" (RL algorithms) to devise better approaches</li>
                  <li>• <strong>Agent agnostic:</strong> Can train any agent through the Interpreter interface</li>
                </ul>
                
                <CodeBlock language="python">
{`# The Coach: Pure training logic ✅
from toolbrain import Brain
from toolbrain.rewards import reward_llm_judge_via_ranking

# The Coach focuses purely on training
coach = Brain(
    agent=athlete,  # The Athlete to train
    reward_func=reward_llm_judge_via_ranking,
    learning_algorithm="GRPO",
    enable_tool_retrieval=True
)

# The Coach orchestrates training without knowing agent internals
coach.train(
    tasks=["Solve math problems", "Write Python code", "Search for information"],
    num_iterations=100
)`}
                </CodeBlock>
              </div>

              {/* The Interpreter */}
              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center mr-4">
                    <span className="text-2xl">🔗</span>
                  </div>
                  <h3 className="text-2xl font-semibold text-purple-400">The Interpreter (Agent Adapter)</h3>
                </div>
                <p className="text-purple-200 mb-4">
                  <strong>Single Responsibility:</strong> Translate between the Coach and Athlete's "languages"
                </p>
                <ul className="text-purple-200 text-sm space-y-2 mb-4">
                  <li>• <strong>Translation layer:</strong> Converts between agent-specific and RL-generic formats</li>
                  <li>• <strong>Execution wrapper:</strong> Provides a standard interface for running any agent</li>
                  <li>• <strong>Trace generator:</strong> Captures execution traces in a format the Coach understands</li>
                  <li>• <strong>Framework bridge:</strong> Enables the Coach to work with different agent frameworks</li>
                  <li>• <strong>Isolation boundary:</strong> Keeps agent and RL concerns completely separate</li>
                </ul>
                
                <CodeBlock language="python">
{`# The Interpreter: Translation layer ✅
from toolbrain.adapters import SmolAgentAdapter

# The Interpreter translates between frameworks
interpreter = SmolAgentAdapter(athlete)

# Coach communicates through the Interpreter
coach = Brain(
    agent=interpreter,  # Uses adapter, not raw agent
    reward_func=reward_llm_judge_via_ranking,
    learning_algorithm="GRPO"
)

# The Interpreter handles all framework-specific details:
# - Converting tasks to agent format
# - Capturing execution traces  
# - Extracting outputs for reward computation
# - Managing agent state between episodes`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </section>

        {/* Mapping to Code */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Mapping the Paradigm to Code</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              The beauty of the Coach-Athlete paradigm is how directly it maps to ToolBrain's class architecture. 
              This isn't just a metaphor - it's the actual design:
            </p>

            <div className="overflow-x-auto mb-6">
              <table className="w-full text-sm text-gray-300 border border-gray-600">
                <thead className="text-xs text-gray-400 uppercase bg-gray-700">
                  <tr>
                    <th className="px-6 py-3 border-r border-gray-600">Paradigm Component</th>
                    <th className="px-6 py-3 border-r border-gray-600">ToolBrain Class</th>
                    <th className="px-6 py-3 border-r border-gray-600">Example Implementation</th>
                    <th className="px-6 py-3">Primary Responsibility</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="bg-gray-800 border-b border-gray-700">
                    <td className="px-6 py-4 font-medium text-green-400 border-r border-gray-600">
                      🏃‍♂️ The Athlete
                    </td>
                    <td className="px-6 py-4 border-r border-gray-600">
                      <code className="text-blue-400">Any Agent</code>
                    </td>
                    <td className="px-6 py-4 border-r border-gray-600">
                      <code className="text-gray-300">smolagents.CodeAgent</code><br/>
                      <code className="text-gray-300">langgraph.Graph</code><br/>
                      <code className="text-gray-300">custom_agent.MyAgent</code>
                    </td>
                    <td className="px-6 py-4 text-green-200">Task execution and domain expertise</td>
                  </tr>
                  <tr className="bg-gray-700 border-b border-gray-600">
                    <td className="px-6 py-4 font-medium text-blue-400 border-r border-gray-600">
                      🧠 The Coach
                    </td>
                    <td className="px-6 py-4 border-r border-gray-600">
                      <code className="text-blue-400">toolbrain.Brain</code>
                    </td>
                    <td className="px-6 py-4 border-r border-gray-600">
                      <code className="text-gray-300">Brain(agent, reward_func)</code>
                    </td>
                    <td className="px-6 py-4 text-blue-200">Training orchestration and RL algorithms</td>
                  </tr>
                  <tr className="bg-gray-800 border-b border-gray-700">
                    <td className="px-6 py-4 font-medium text-purple-400 border-r border-gray-600">
                      🔗 The Interpreter
                    </td>
                    <td className="px-6 py-4 border-r border-gray-600">
                      <code className="text-blue-400">toolbrain.adapters.*</code>
                    </td>
                    <td className="px-6 py-4 border-r border-gray-600">
                      <code className="text-gray-300">SmolAgentAdapter</code><br/>
                      <code className="text-gray-300">LangGraphAdapter</code><br/>
                      <code className="text-gray-300">CustomAdapter</code>
                    </td>
                    <td className="px-6 py-4 text-purple-200">Framework translation and trace capture</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-2">🎯 Complete Example: All Components Working Together</h4>
              <CodeBlock language="python">
{`#!/usr/bin/env python3
"""
Complete Coach-Athlete Paradigm Example
Shows how all three components work together seamlessly
"""

from smolagents import CodeAgent
from toolbrain import Brain
from toolbrain.adapters import SmolAgentAdapter
from toolbrain.rewards import reward_llm_judge_via_ranking

# Step 1: Create The Athlete (pure task execution)
athlete = CodeAgent(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    tools=["python_interpreter", "web_search", "file_reader"]
)

# Step 2: Create The Interpreter (translation layer)
interpreter = SmolAgentAdapter(athlete)

# Step 3: Create The Coach (training orchestration)
coach = Brain(
    agent=interpreter,  # Coach works through Interpreter
    reward_func=reward_llm_judge_via_ranking,
    learning_algorithm="GRPO",
    enable_tool_retrieval=True
)

# Step 4: Training (Coach-Athlete paradigm in action)
tasks = [
    "Calculate the factorial of 15",
    "Find the current price of Bitcoin",
    "Read the contents of data.csv and compute statistics"
]

# The Coach trains the Athlete through the Interpreter
coach.train(tasks=tasks, num_iterations=50)

# The paradigm benefits:
# - Athlete focuses purely on task execution
# - Coach handles all RL complexity
# - Interpreter bridges different frameworks
# - Each component can be modified independently`}
              </CodeBlock>
            </div>
          </div>
        </section>

        {/* Why This Matters */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Why This Matters: The Benefits</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              The Coach-Athlete paradigm delivers concrete benefits for both users and the framework itself:
            </p>

            <div className="space-y-8">
              {/* For Users */}
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-6">
                <h3 className="text-2xl font-semibold text-green-400 mb-4">👥 For Users: Radical Simplicity</h3>
                <p className="text-green-200 mb-4">
                  Users can focus on what they do best - creating effective agents - without needing deep RL expertise.
                </p>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold text-green-300 mb-2">🎯 What Users Focus On</h4>
                    <ul className="text-green-200 text-sm space-y-1">
                      <li>• Creating a good "Athlete" (the agent)</li>
                      <li>• Defining "rules of the game" (reward function)</li>
                      <li>• Providing example tasks</li>
                      <li>• Domain-specific tool selection</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold text-green-300 mb-2">🚫 What Users Don't Need to Know</h4>
                    <ul className="text-green-200 text-sm space-y-1">
                      <li>• RL algorithm implementation details</li>
                      <li>• Gradient computation and backpropagation</li>
                      <li>• Training loop optimization</li>
                      <li>• Policy network architectures</li>
                    </ul>
                  </div>
                </div>

                <div className="mt-4 p-4 bg-gray-700 rounded-lg">
                  <h4 className="font-semibold text-green-300 mb-2">📊 User Experience Comparison</h4>
                  <div className="space-y-3">
                    <div className="flex items-center gap-4">
                      <span className="text-red-400 w-16">Before:</span>
                      <span className="text-gray-300 text-sm">"I need to learn GRPO, PPO, reward modeling, and policy gradients to train my agent"</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="text-green-400 w-16">After:</span>
                      <span className="text-gray-300 text-sm">"I need to create my agent and define what 'good performance' looks like"</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* For Framework */}
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-6">
                <h3 className="text-2xl font-semibold text-blue-400 mb-4">🔧 For the Framework: Unlimited Extensibility</h3>
                <p className="text-blue-200 mb-4">
                  Adding support for new agent frameworks requires zero changes to the core training logic.
                </p>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-blue-300 mb-2">🔌 Adding New Agent Frameworks</h4>
                    <p className="text-blue-200 text-sm mb-3">
                      To support a new agent framework (e.g., LangGraph, CrewAI, AutoGen), we simply create a new Interpreter:
                    </p>
                    
                    <CodeBlock language="python">
{`# Adding LangGraph support - no Brain changes needed!
class LangGraphAdapter(AgentAdapter):
    def __init__(self, graph):
        self.graph = graph
    
    def run(self, task: str) -> ExecutionTrace:
        # Convert task to LangGraph format
        graph_input = {"input": task}
        
        # Run the graph and capture execution
        result = self.graph.invoke(graph_input)
        
        # Convert to standard ExecutionTrace format
        return ExecutionTrace(
            steps=self._extract_steps(result),
            final_output=result["output"],
            tools_used=self._extract_tools(result)
        )

# Now Brain can train LangGraph agents too!
langgraph_agent = create_langgraph()
adapter = LangGraphAdapter(langgraph_agent)
brain = Brain(agent=adapter, reward_func=my_reward)
brain.train(tasks=my_tasks)`}
                    </CodeBlock>
                  </div>

                  <div className="bg-gray-700 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-300 mb-2">🎯 Framework Support Matrix</h4>
                    <div className="grid md:grid-cols-3 gap-4 text-sm">
                      <div>
                        <h5 className="text-green-400 mb-1">✅ Currently Supported</h5>
                        <ul className="text-gray-300 space-y-1">
                          <li>• SmolAgents</li>
                          <li>• Custom agents</li>
                        </ul>
                      </div>
                      <div>
                        <h5 className="text-yellow-400 mb-1">🚧 Easy to Add</h5>
                        <ul className="text-gray-300 space-y-1">
                          <li>• LangGraph</li>
                          <li>• LangChain</li>
                          <li>• CrewAI</li>
                          <li>• AutoGen</li>
                        </ul>
                      </div>
                      <div>
                        <h5 className="text-blue-400 mb-1">💡 Future Possibilities</h5>
                        <ul className="text-gray-300 space-y-1">
                          <li>• Web agents</li>
                          <li>• Robotics agents</li>
                          <li>• Multi-modal agents</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Comparison to Actor-Learner */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Comparison to Actor-Learner Architecture</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              While the Coach-Athlete paradigm shares similarities with the classic Actor-Learner model from RL literature, 
              our separation boundary is strategically different and optimized for rapid, user-centric development:
            </p>

            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-yellow-400 mb-3">🎭 Traditional Actor-Learner</h3>
                <ul className="text-yellow-200 text-sm space-y-2 mb-4">
                  <li>• <strong>Actor:</strong> Policy network that selects actions</li>
                  <li>• <strong>Learner:</strong> Updates policy based on collected experience</li>
                  <li>• <strong>Boundary:</strong> Between action selection and policy updates</li>
                  <li>• <strong>Focus:</strong> Optimizing RL algorithm performance</li>
                </ul>
                <div className="bg-gray-700 rounded p-3">
                  <p className="text-yellow-200 text-xs">
                    <strong>Limitation:</strong> Still requires deep RL knowledge to implement the Actor component properly.
                  </p>
                </div>
              </div>

              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-blue-400 mb-3">🏃‍♂️ ToolBrain Coach-Athlete</h3>
                <ul className="text-blue-200 text-sm space-y-2 mb-4">
                  <li>• <strong>Athlete:</strong> Complete task-solving agent (any framework)</li>
                  <li>• <strong>Coach:</strong> Complete training orchestration system</li>
                  <li>• <strong>Boundary:</strong> Between user-facing task execution and RL training</li>
                  <li>• <strong>Focus:</strong> Maximizing user productivity and framework flexibility</li>
                </ul>
                <div className="bg-gray-700 rounded p-3">
                  <p className="text-blue-200 text-xs">
                    <strong>Advantage:</strong> Users can focus entirely on their domain expertise without any RL knowledge.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="font-semibold text-gray-300 mb-3">🔍 Key Architectural Differences</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-gray-300">
                  <thead className="text-xs text-gray-400 uppercase">
                    <tr>
                      <th className="px-4 py-2 text-left">Aspect</th>
                      <th className="px-4 py-2 text-left">Actor-Learner</th>
                      <th className="px-4 py-2 text-left">Coach-Athlete</th>
                    </tr>
                  </thead>
                  <tbody className="space-y-1">
                    <tr>
                      <td className="px-4 py-2 font-medium">Separation Point</td>
                      <td className="px-4 py-2 text-yellow-200">Action selection vs learning</td>
                      <td className="px-4 py-2 text-blue-200">Task execution vs training</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-2 font-medium">User Interface</td>
                      <td className="px-4 py-2 text-yellow-200">Policy network design</td>
                      <td className="px-4 py-2 text-blue-200">Agent creation + reward definition</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-2 font-medium">RL Knowledge Required</td>
                      <td className="px-4 py-2 text-yellow-200">High (policy networks, etc.)</td>
                      <td className="px-4 py-2 text-blue-200">None (handled by Coach)</td>
                    </tr>
                    <tr>
                      <td className="px-4 py-2 font-medium">Framework Support</td>
                      <td className="px-4 py-2 text-yellow-200">RL-specific implementations</td>
                      <td className="px-4 py-2 text-blue-200">Any agent framework via adapters</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="mt-4 p-4 bg-green-900/20 border border-green-600 rounded-lg">
              <h4 className="font-semibold text-green-400 mb-2">🎯 Why Our Boundary Matters</h4>
              <p className="text-green-200 text-sm">
                By placing the separation boundary between <em>user-facing orchestration</em> and <em>task execution</em>, 
                rather than between <em>action selection</em> and <em>learning</em>, we create a system that's both more 
                accessible to developers and more adaptable to different agent frameworks. Users work with familiar 
                agent concepts, while the framework handles all RL complexity behind the scenes.
              </p>
            </div>
          </div>
        </section>

        {/* Conclusion */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The Paradigm in Practice</h2>
          <div className="bg-gray-800 rounded-lg p-6">
            <p className="text-gray-300 mb-4">
              The Coach-Athlete paradigm isn't just a design philosophy - it's a practical architecture that delivers 
              real benefits every day:
            </p>

            <div className="grid md:grid-cols-3 gap-6 mb-6">
              <div className="bg-green-900/20 border border-green-600 rounded-lg p-4 text-center">
                <div className="text-3xl mb-2">⚡</div>
                <h3 className="font-semibold text-green-400 mb-2">Faster Development</h3>
                <p className="text-green-200 text-sm">Go from idea to trained agent in hours, not weeks</p>
              </div>
              
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4 text-center">
                <div className="text-3xl mb-2">🔧</div>
                <h3 className="font-semibold text-blue-400 mb-2">Lower Barriers</h3>
                <p className="text-blue-200 text-sm">No RL expertise required - focus on your domain</p>
              </div>
              
              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-4 text-center">
                <div className="text-3xl mb-2">🌐</div>
                <h3 className="font-semibold text-purple-400 mb-2">Universal Support</h3>
                <p className="text-purple-200 text-sm">Works with any agent framework via adapters</p>
              </div>
            </div>

            <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-2">🎯 Ready to Get Started?</h4>
              <p className="text-blue-200 text-sm mb-3">
                The Coach-Athlete paradigm makes RL-powered agent training as simple as:
              </p>
              <ol className="text-blue-200 text-sm space-y-1 list-decimal list-inside">
                <li>Create your agent (the Athlete) using your preferred framework</li>
                <li>Define what good performance looks like (the reward function)</li>
                <li>Let the Coach handle all the training complexity</li>
              </ol>
              <p className="text-blue-200 text-sm mt-3">
                Check out our <strong>Quickstart Guide</strong> to see this paradigm in action with a complete working example.
              </p>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}