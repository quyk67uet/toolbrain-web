'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import Image from 'next/image';

export default function UnifiedAPI() {
  const basicCode = `from toolbrain import Brain, reward_exact_match
from smolagents import CodeAgent, TransformersModel

# 1. Define your agent and reward function
agent = CodeAgent(model=TransformersModel("Qwen/3B"), tools=[...])
reward_func = reward_exact_match

# 2. Initialize the Brain and train
brain = Brain(agent=agent, reward_func=reward_func)
brain.train(dataset=[{"query": "...", "gold_answer": "..."}])`;

  const advancedCode = `# The Brain orchestrates all key features through one API

# 1. Define advanced components
reward_func = reward_llm_judge_via_ranking
retriever = ToolRetriever(tools=all_available_tools)
agent = CodeAgent(model="Qwen/3B", tools=[]) # Tools will be handled by Brain

# 2. Configure the Brain with all components and strategies
brain = Brain(
    agent=agent, 
    reward_func=reward_func,
    tool_retriever=retriever, 
    learning_algorithm="DPO"
)

# 3. Use advanced features to prepare and execute training
training_tasks = brain.generate_training_examples("Master finance APIs")
brain.distill(dataset=training_tasks, teacher_model_id="GPT-4-Turbo")
brain.train(dataset=training_tasks)`;

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        {/* HEADER SECTION */}
        <div className="text-center mb-16">
          <h1 className="text-5xl font-bold text-[#E6EDF3] mb-6">
            Unified API & Architecture
          </h1>
          <p className="text-xl text-gray-400 max-w-4xl mx-auto leading-relaxed">
            The heart of ToolBrain is the Brain class. It provides a single, unified API that handles 
            the entire complex Reinforcement Learning workflow. At its simplest, training an agent 
            takes just a few lines of code.
          </p>
        </div>

        {/* PHẦN 1: THE SIMPLE PATH */}
        <div className="mb-20">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            The Simple Path: A Basic Training Run
          </h2>
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            This example shows the minimal code required to train a standard smolagents.CodeAgent.
          </p>
          
          <CodeBlock language="python" filename="basic_training.py">
            {basicCode}
          </CodeBlock>
        </div>

        {/* PHẦN 2: THE POWERFUL PATH */}
        <div className="mb-20">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            The Powerful Path: A Complete Workflow
          </h2>
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            Beyond the basics, the Brain API serves as a unified interface to orchestrate all of ToolBrain's 
            advanced features. The conceptual code below demonstrates how you can combine multiple capabilities 
            into a single, powerful workflow.
          </p>
          
          <CodeBlock language="python" filename="advanced_workflow.py">
            {advancedCode}
          </CodeBlock>
        </div>

        {/* PHẦN 3: THE MAGIC BEHIND THE SCENES - AGENT ADAPTER */}
        <div className="mb-20">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            How It Works: The Agent Adapter
          </h2>
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            You might wonder how the Brain can work with different agent libraries like smolagents, 
            LangChain, or LlamaIndex without requiring you to change your agent code. The magic is in 
            the Agent Adapter.
          </p>
          <p className="text-lg text-gray-400 mb-12 leading-relaxed">
            The Adapter is an internal component, automatically selected by the Brain, that acts as a 
            universal translator. It wraps your agent, observes its actions, and translates its unique 
            internal memory into a standardized Execution Trace that our RL algorithms can understand.
          </p>

          {/* VISUAL DIAGRAM */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-16 mb-12 relative overflow-hidden">
            {/* Background pattern */}
            <div className="absolute inset-0 opacity-5">
              <svg className="w-full h-full" viewBox="0 0 100 100">
                <defs>
                  <pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse">
                    <path d="M 10 0 L 0 0 0 10" fill="none" stroke="currentColor" strokeWidth="0.5"/>
                  </pattern>
                </defs>
                <rect width="100" height="100" fill="url(#grid)" />
              </svg>
            </div>

            <div className="relative flex flex-col xl:flex-row items-center justify-center gap-12 xl:gap-16">
              
              {/* LEFT: Multiple Agent Frameworks */}
              <div className="flex flex-col items-center">
                <h3 className="text-2xl font-bold text-[#E6EDF3] mb-8 text-center">Different Agent Frameworks</h3>
                
                <div className="space-y-6">
                  {/* SmolaAgents */}
                  <div className="relative group">
                    <div className="flex items-center gap-6 bg-gradient-to-r from-[#0D1117] to-[#1C2128] border-2 border-[#30363D] rounded-xl p-6 hover:border-[#F85149]/50 transition-all duration-300 hover:scale-105 min-w-[280px]">
                      <Image src="/smolagents.png" alt="SmolaAgents" width={48} height={48} className="rounded-lg" />
                      <div className="flex-1">
                        <span className="text-[#E6EDF3] font-bold text-lg">smolagents</span>
                        <div className="flex items-center gap-2 mt-2">
                          <div className="flex space-x-1">
                            <div className="w-3 h-3 bg-red-400 rounded-full animate-pulse"></div>
                            <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                            <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                          </div>
                          <span className="text-xs text-gray-400">Different protocols</span>
                        </div>
                      </div>
                    </div>
                    {/* Connection line to adapter */}
                    <div className="hidden xl:block absolute top-1/2 -right-8 w-8 h-0.5 bg-gradient-to-r from-[#F85149] to-[#58A6FF] transform -translate-y-1/2"></div>
                  </div>
                  
                  {/* LangChain */}
                  <div className="relative group">
                    <div className="flex items-center gap-6 bg-gradient-to-r from-[#0D1117] to-[#1C2128] border-2 border-[#30363D] rounded-xl p-6 hover:border-[#7C3AED]/50 transition-all duration-300 hover:scale-105 min-w-[280px]">
                      <Image src="/lang.png" alt="LangChain" width={48} height={48} className="rounded-lg" />
                      <div className="flex-1">
                        <span className="text-[#E6EDF3] font-bold text-lg">LangChain / LangGraph</span>
                        <div className="flex items-center gap-2 mt-2">
                          <div className="flex space-x-1">
                            <div className="w-3 h-3 bg-yellow-400 rounded-full animate-pulse"></div>
                            <div className="w-3 h-3 bg-purple-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                            <div className="w-3 h-3 bg-pink-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                          </div>
                          <span className="text-xs text-gray-400">Different protocols</span>
                        </div>
                      </div>
                    </div>
                    {/* Connection line to adapter */}
                    <div className="hidden xl:block absolute top-1/2 -right-8 w-8 h-0.5 bg-gradient-to-r from-[#7C3AED] to-[#58A6FF] transform -translate-y-1/2"></div>
                  </div>
                  
                  {/* LlamaIndex */}
                  <div className="relative group">
                    <div className="flex items-center gap-6 bg-gradient-to-r from-[#0D1117] to-[#1C2128] border-2 border-[#30363D] rounded-xl p-6 hover:border-[#3FB950]/50 transition-all duration-300 hover:scale-105 min-w-[280px]">
                      <Image src="/llama.png" alt="LlamaIndex" width={48} height={48} className="rounded-lg" />
                      <div className="flex-1">
                        <span className="text-[#E6EDF3] font-bold text-lg">LlamaIndex</span>
                        <div className="flex items-center gap-2 mt-2">
                          <div className="flex space-x-1">
                            <div className="w-3 h-3 bg-orange-400 rounded-full animate-pulse"></div>
                            <div className="w-3 h-3 bg-teal-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                            <div className="w-3 h-3 bg-indigo-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                          </div>
                          <span className="text-xs text-gray-400">Different protocols</span>
                        </div>
                      </div>
                    </div>
                    {/* Connection line to adapter */}
                    <div className="hidden xl:block absolute top-1/2 -right-8 w-8 h-0.5 bg-gradient-to-r from-[#3FB950] to-[#58A6FF] transform -translate-y-1/2"></div>
                  </div>
                </div>
              </div>

              {/* MIDDLE: Adapter */}
              <div className="relative flex flex-col items-center">
                {/* Main Arrow (mobile) */}
                <div className="xl:hidden mb-6">
                  <svg className="w-12 h-16 text-[#58A6FF] animate-bounce" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M1 13.025l2.828-2.847 6.176 6.176v-16.354h3.992v16.354l6.176-6.176 2.828 2.847-11 10.975z"/>
                  </svg>
                </div>
                
                {/* Adapter Box */}
                <div className="relative">
                  {/* Glow effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-[#58A6FF]/30 to-[#7C3AED]/30 rounded-2xl blur-xl"></div>
                  
                  <div className="relative bg-gradient-to-br from-[#58A6FF]/20 to-[#7C3AED]/20 border-3 border-[#58A6FF] rounded-2xl p-8 backdrop-blur-sm">
                    <div className="text-center">
                      <div className="text-5xl mb-4 animate-spin-slow">🔄</div>
                      <h4 className="text-2xl font-bold text-[#58A6FF] mb-2">ToolBrain</h4>
                      <h4 className="text-2xl font-bold text-[#7C3AED]">Adapter</h4>
                      <div className="mt-4 px-4 py-2 bg-[#0D1117] rounded-lg border border-[#30363D]">
                        <p className="text-sm text-[#58A6FF] font-semibold">Universal</p>
                        <p className="text-sm text-[#7C3AED] font-semibold">Translator</p>
                      </div>
                      <div className="mt-4 flex justify-center">
                        <div className="w-4 h-4 bg-[#58A6FF] rounded-full animate-pulse"></div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Arrow to Brain (mobile) */}
                <div className="xl:hidden mt-6">
                  <svg className="w-12 h-16 text-[#58A6FF] animate-bounce" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M1 13.025l2.828-2.847 6.176 6.176v-16.354h3.992v16.354l6.176-6.176 2.828 2.847-11 10.975z"/>
                  </svg>
                </div>
              </div>

              {/* RIGHT: Unified ToolBrain */}
              <div className="relative flex flex-col items-center">
                <h3 className="text-2xl font-bold text-[#E6EDF3] mb-8 text-center">Unified Interface</h3>
                
                <div className="relative">
                  {/* Glow effect */}
                  <div className="absolute inset-0 bg-gradient-to-r from-[#58A6FF]/40 to-[#3FB950]/40 rounded-2xl blur-2xl"></div>
                  
                  <div className="relative bg-gradient-to-br from-[#58A6FF]/15 to-[#3FB950]/15 border-3 border-[#58A6FF] rounded-2xl p-10 hover:scale-105 transition-all duration-300 min-w-[280px]">
                    <div className="flex flex-col items-center space-y-6">
                      <div className="relative">
                        <Image src="/toolbrain.png" alt="ToolBrain" width={80} height={80} className="rounded-xl" />
                        <div className="absolute -top-2 -right-2 w-6 h-6 bg-[#3FB950] rounded-full flex items-center justify-center">
                          <span className="text-white text-xs font-bold">✓</span>
                        </div>
                      </div>
                      <h4 className="text-3xl font-bold text-[#58A6FF]">ToolBrain</h4>
                      <div className="text-center bg-[#0D1117] rounded-lg px-4 py-3 border border-[#30363D] w-full">
                        <p className="text-lg font-semibold text-[#3FB950]">Single Standardized</p>
                        <p className="text-lg font-semibold text-[#58A6FF]">Brain API</p>
                      </div>
                      <div className="flex space-x-2">
                        <div className="w-3 h-3 bg-[#58A6FF] rounded-full animate-pulse"></div>
                        <div className="w-3 h-3 bg-[#3FB950] rounded-full animate-pulse" style={{animationDelay: '0.3s'}}></div>
                        <div className="w-3 h-3 bg-[#7C3AED] rounded-full animate-pulse" style={{animationDelay: '0.6s'}}></div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Connection line from adapter */}
                  <div className="hidden xl:block absolute top-1/2 -left-8 w-8 h-0.5 bg-gradient-to-r from-[#58A6FF] to-[#3FB950] transform -translate-y-1/2"></div>
                </div>
              </div>
            </div>

            {/* Animated background particles */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-[#58A6FF] rounded-full opacity-60 animate-ping"></div>
              <div className="absolute top-3/4 right-1/4 w-1 h-1 bg-[#7C3AED] rounded-full opacity-40 animate-ping" style={{animationDelay: '1s'}}></div>
              <div className="absolute bottom-1/4 left-1/3 w-1.5 h-1.5 bg-[#3FB950] rounded-full opacity-50 animate-ping" style={{animationDelay: '2s'}}></div>
            </div>
          </div>

          {/* SUPPORT & ROADMAP */}
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-8">
            <h3 className="text-2xl font-bold text-[#E6EDF3] mb-6">
              Current Support & Future Roadmap
            </h3>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h4 className="text-lg font-semibold text-[#3FB950] mb-4 flex items-center gap-2">
                  <span className="text-xl">✅</span>
                  Currently Supported
                </h4>
                <ul className="space-y-2 text-gray-300">
                  <li className="flex items-center gap-3">
                    <Image src="/smolagents.png" alt="SmolaAgents" width={20} height={20} className="rounded" />
                    smolagents
                  </li>
                  <li className="flex items-center gap-3">
                    <Image src="/lang.png" alt="LangChain" width={20} height={20} className="rounded" />
                    LangChain / LangGraph
                  </li>
                  <li className="flex items-center gap-3">
                    <Image src="/llama.png" alt="LlamaIndex" width={20} height={20} className="rounded" />
                    LlamaIndex
                  </li>
                </ul>
              </div>
              
              <div>
                <h4 className="text-lg font-semibold text-[#58A6FF] mb-4 flex items-center gap-2">
                  <span className="text-xl">🚀</span>
                  Coming Soon
                </h4>
                <ul className="space-y-2 text-gray-300">
                  <li className="flex items-center gap-3">
                    <Image src="/autogen.jpg" alt="AutoGen" width={20} height={20} className="rounded" />
                    AutoGen
                  </li>
                  <li className="flex items-center gap-3">
                    <Image src="/crew.png" alt="CrewAI" width={20} height={20} className="rounded" />
                    CrewAI
                  </li>
                  <li className="flex items-center gap-3">
                    <div className="w-5 h-5 bg-gradient-to-r from-green-400 to-blue-400 rounded flex items-center justify-center text-xs font-bold text-white">+</div>
                    Custom Adapters
                  </li>
                </ul>
              </div>
            </div>
            
            <p className="text-gray-400 mt-6 text-sm leading-relaxed">
              The modular design also allows advanced users to easily create adapters for their own custom agents.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}