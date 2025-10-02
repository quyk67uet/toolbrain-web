'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function IntelligentToolRetrieval() {
  const toolRetrievalCode = `from toolbrain import Brain, ToolRetriever
from smolagents import CodeAgent
from my_tools import add, multiply, divide, subtract, send_email, search_web

# 1. Define your agent with the FULL library of tools
all_tools = [add, multiply, divide, subtract, send_email, search_web]
agent = CodeAgent(
    model=UnslothModel(model_id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=all_tools
)

# 2. Create and configure the ToolRetriever
# It uses a separate, powerful LLM to perform the selection.
tool_retriever = ToolRetriever(
    llm_model="gpt-4o-mini",
    # You can provide guidelines to steer the retriever's behavior
    retrieval_guidelines="Select only tools needed for mathematical calculations"
)

# 3. Pass the retriever to the Brain
# The Brain will now automatically manage the tool selection process.
brain = Brain(
    agent=agent,
    reward_func=my_reward_function,
    tool_retriever=tool_retriever, # Enable and configure in one step
    learning_algorithm="GRPO"
)

# Now, when you run brain.train(), the retriever will be used for each query.
brain.train(dataset=...)`;

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        {/* PHẦN 1: THE PROBLEM - "TOO MANY TOOLS" */}
        <div className="text-center mb-20">
          <h1 className="text-5xl font-bold text-[#E6EDF3] mb-8">
            Intelligent Tool Retrieval
          </h1>
          
          <div className="max-w-4xl mx-auto space-y-6">
            <p className="text-xl text-gray-400 leading-relaxed">
              In real-world applications, an agent might have access to dozens or even hundreds of tools. 
              Providing all of them to the language model in every single prompt is highly inefficient and often counterproductive.
            </p>
            
            <div className="grid md:grid-cols-3 gap-6 my-12">
              <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-xl p-6">
                <div className="text-3xl mb-4 text-center">🎯</div>
                <h3 className="text-lg font-bold text-[#58A6FF] mb-3 text-center">Smart Context Management</h3>
                <p className="text-gray-300 text-sm text-center leading-relaxed">
                  Automatically optimizes context window usage by selecting only relevant tools
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-[#7C3AED]/10 to-[#6366F1]/10 border border-[#7C3AED]/30 rounded-xl p-6">
                <div className="text-3xl mb-4 text-center">🧠</div>
                <h3 className="text-lg font-bold text-[#7C3AED] mb-3 text-center">Intelligent Selection</h3>
                <p className="text-gray-300 text-sm text-center leading-relaxed">
                  LLM-powered analysis ensures agents get exactly the right tools for each task
                </p>
              </div>
              
              <div className="bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-xl p-6">
                <div className="text-3xl mb-4 text-center">⚡</div>
                <h3 className="text-lg font-bold text-[#3FB950] mb-3 text-center">Enhanced Performance</h3>
                <p className="text-gray-300 text-sm text-center leading-relaxed">
                  Focused tool sets lead to faster training and more accurate execution
                </p>
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-[#58A6FF]/10 to-[#3FB950]/10 border border-[#58A6FF]/30 rounded-xl p-6">
              <div className="flex items-center justify-center gap-3 mb-3">
                <div className="text-2xl">🔍</div>
                <h3 className="text-xl font-bold text-[#58A6FF]">ToolBrain's Solution</h3>
              </div>
              <p className="text-xl font-semibold text-center text-[#E6EDF3]">
                ToolBrain solves this with a built-in, intelligent Tool Retriever.
              </p>
            </div>
          </div>
        </div>

        {/* Scale Showcase */}
        <div className="mb-20">
          <h2 className="text-3xl font-bold text-[#E6EDF3] text-center mb-8">
            From Small to Large: Scaling Your Tool Library
          </h2>
          
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-xl p-8">
              <div className="text-center mb-6">
                <div className="text-4xl mb-4">�</div>
                <h3 className="text-2xl font-bold text-[#58A6FF]">Traditional Approach</h3>
                <p className="text-gray-400 text-sm mt-2">Static tool selection</p>
              </div>
              
              <div className="space-y-4">
                <div className="bg-[#0D1117] rounded-lg p-4">
                  <p className="text-[#58A6FF] text-sm font-semibold mb-2">Query: "What is 5+7?"</p>
                  <p className="text-gray-300 text-sm mb-2">All tools provided to agent:</p>
                  <div className="flex flex-wrap gap-2">
                    <span className="bg-[#58A6FF]/20 text-[#58A6FF] px-2 py-1 rounded text-xs">add</span>
                    <span className="bg-[#58A6FF]/20 text-[#58A6FF] px-2 py-1 rounded text-xs">multiply</span>
                    <span className="bg-[#58A6FF]/20 text-[#58A6FF] px-2 py-1 rounded text-xs">divide</span>
                    <span className="bg-[#58A6FF]/20 text-[#58A6FF] px-2 py-1 rounded text-xs">subtract</span>
                    <span className="bg-[#58A6FF]/20 text-[#58A6FF] px-2 py-1 rounded text-xs">send_email</span>
                    <span className="bg-[#58A6FF]/20 text-[#58A6FF] px-2 py-1 rounded text-xs">search_web</span>
                    <span className="bg-[#58A6FF]/20 text-[#58A6FF] px-2 py-1 rounded text-xs">+ many more...</span>
                  </div>
                </div>
                <div className="text-center text-[#58A6FF] text-sm">
                  Works for small tool sets, but doesn't scale
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-xl p-8">
              <div className="text-center mb-6">
                <div className="text-4xl mb-4">🚀</div>
                <h3 className="text-2xl font-bold text-[#3FB950]">ToolBrain's Smart Retrieval</h3>
                <p className="text-gray-400 text-sm mt-2">Dynamic, intelligent selection</p>
              </div>
              
              <div className="space-y-4">
                <div className="bg-[#0D1117] rounded-lg p-4">
                  <p className="text-[#58A6FF] text-sm font-semibold mb-2">Query: "What is 5+7?"</p>
                  <p className="text-gray-300 text-sm mb-2">Intelligently selected tools:</p>
                  <div className="flex flex-wrap gap-2">
                    <span className="bg-[#3FB950]/20 text-[#3FB950] px-2 py-1 rounded text-xs">add</span>
                    <span className="bg-[#3FB950]/20 text-[#3FB950] px-2 py-1 rounded text-xs">subtract</span>
                  </div>
                </div>
                <div className="text-center text-[#3FB950] text-sm">
                  Scales effortlessly to hundreds of tools
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* PHẦN 2: THE SOLUTION - TWO-PASS APPROACH */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] text-center mb-8">
            How It Works: A Dynamic, Two-Pass Approach
          </h2>
          
          <p className="text-lg text-gray-400 mb-12 text-center max-w-3xl mx-auto leading-relaxed">
            Instead of statically providing all tools, the Brain uses the Tool Retriever to perform a dynamic, 
            just-in-time selection process for each task.
          </p>

          {/* VISUAL PIPELINE */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12">
            
            {/* Mobile Layout */}
            <div className="flex flex-col lg:hidden space-y-12">
              
              {/* User Query */}
              <div className="text-center">
                <div className="bg-gradient-to-br from-[#58A6FF]/20 to-[#4A90E2]/20 border-2 border-[#58A6FF] rounded-xl p-6">
                  <div className="text-4xl mb-4">👤</div>
                  <h3 className="text-xl font-bold text-[#58A6FF] mb-3">User Query</h3>
                  <div className="bg-[#0D1117] rounded-lg p-3">
                    <p className="text-[#E6EDF3] font-medium">"What is 5+7?"</p>
                  </div>
                </div>
              </div>

              {/* Retrieval Pass */}
              <div className="text-center">
                <div className="bg-gradient-to-br from-[#7C3AED]/20 to-[#6366F1]/20 border-2 border-[#7C3AED] rounded-xl p-6">
                  <div className="flex items-center justify-center gap-4 mb-4">
                    <div className="text-3xl">🔍</div>
                    <div className="text-xl">→</div>
                    <div className="grid grid-cols-3 gap-1">
                      <span className="text-sm">➕</span>
                      <span className="text-sm">✖️</span>
                      <span className="text-sm">➗</span>
                      <span className="text-sm">➖</span>
                      <span className="text-sm">📧</span>
                      <span className="text-sm">🌐</span>
                    </div>
                    <div className="text-xl">→</div>
                    <div className="flex gap-2">
                      <span className="text-lg">➕</span>
                      <span className="text-lg">➖</span>
                    </div>
                  </div>
                  <h3 className="text-xl font-bold text-[#7C3AED] mb-3">1. Retrieval Pass</h3>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    The ToolRetriever uses a powerful LLM to analyze the user's query and select only the most relevant tools from the full library.
                  </p>
                </div>
              </div>

              {/* Execution Pass */}
              <div className="text-center">
                <div className="bg-gradient-to-br from-[#3FB950]/20 to-[#10B981]/20 border-2 border-[#3FB950] rounded-xl p-6">
                  <div className="flex items-center justify-center gap-4 mb-4">
                    <div className="flex gap-2">
                      <span className="text-lg">➕</span>
                      <span className="text-lg">➖</span>
                    </div>
                    <div className="text-xl">→</div>
                    <div className="text-3xl">🤖</div>
                    <div className="text-xl">→</div>
                    <div className="text-2xl">✅</div>
                  </div>
                  <h3 className="text-xl font-bold text-[#3FB950] mb-3">2. Execution Pass</h3>
                  <p className="text-gray-300 text-sm leading-relaxed">
                    The Brain then provides the agent with this small, focused set of tools to perform its task.
                  </p>
                </div>
              </div>
            </div>

            {/* Desktop Layout */}
            <div className="hidden lg:block">
              <div className="flex items-center justify-center gap-8">
                
                {/* User Query */}
                <div className="flex-shrink-0">
                  <div className="bg-gradient-to-br from-[#58A6FF]/20 to-[#4A90E2]/20 border-2 border-[#58A6FF] rounded-xl p-4 text-center w-32">
                    <div className="text-3xl mb-2">👤</div>
                    <h4 className="text-sm font-bold text-[#58A6FF] mb-2">User Query</h4>
                    <div className="bg-[#0D1117] rounded p-2">
                      <p className="text-xs text-[#E6EDF3]">"What is 5+7?"</p>
                    </div>
                  </div>
                </div>

                {/* Arrow 1 */}
                <div className="flex justify-center">
                  <svg className="w-8 h-6 text-[#58A6FF]" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M13.025 1l-2.847 2.828 6.176 6.176h-16.354v3.992h16.354l-6.176 6.176 2.847 2.828 10.975-11z"/>
                  </svg>
                </div>

                {/* Retrieval Process */}
                <div className="flex-1">
                  <div className="bg-gradient-to-br from-[#7C3AED]/20 to-[#6366F1]/20 border-2 border-[#7C3AED] rounded-xl p-6">
                    <h3 className="text-lg font-bold text-[#7C3AED] mb-4 text-center">1. Retrieval Pass</h3>
                    
                    <div className="flex items-center justify-center gap-4 mb-4">
                      <div className="text-2xl">🔍</div>
                      <div className="flex flex-col items-center">
                        <p className="text-xs text-gray-400 mb-2">Full Tool Library</p>
                        <div className="grid grid-cols-3 gap-1 bg-[#0D1117] p-2 rounded">
                          <span className="text-xs">➕</span>
                          <span className="text-xs">✖️</span>
                          <span className="text-xs">➗</span>
                          <span className="text-xs">➖</span>
                          <span className="text-xs">📧</span>
                          <span className="text-xs">🌐</span>
                        </div>
                      </div>
                      <div className="text-lg">→</div>
                      <div className="flex flex-col items-center">
                        <p className="text-xs text-gray-400 mb-2">Selected Tools</p>
                        <div className="flex gap-2 bg-[#0D1117] p-2 rounded">
                          <span className="text-sm">➕</span>
                          <span className="text-sm">➖</span>
                        </div>
                      </div>
                    </div>
                    
                    <p className="text-xs text-gray-300 text-center leading-relaxed">
                      LLM analyzes query and selects relevant tools
                    </p>
                  </div>
                </div>

                {/* Arrow 2 */}
                <div className="flex justify-center">
                  <svg className="w-8 h-6 text-[#58A6FF]" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M13.025 1l-2.847 2.828 6.176 6.176h-16.354v3.992h16.354l-6.176 6.176 2.847 2.828 10.975-11z"/>
                  </svg>
                </div>

                {/* Execution Pass */}
                <div className="flex-shrink-0">
                  <div className="bg-gradient-to-br from-[#3FB950]/20 to-[#10B981]/20 border-2 border-[#3FB950] rounded-xl p-4 text-center w-32">
                    <div className="text-3xl mb-2">🤖</div>
                    <h4 className="text-sm font-bold text-[#3FB950] mb-2">2. Execution</h4>
                    <div className="flex gap-1 justify-center mb-2">
                      <span className="text-sm">➕</span>
                      <span className="text-sm">➖</span>
                    </div>
                    <div className="text-lg">✅</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#3FB950]/10 border border-[#3FB950]/30 rounded-xl p-6 mt-8">
            <div className="flex items-center gap-3 mb-3">
              <div className="text-2xl">🎯</div>
              <h3 className="text-xl font-bold text-[#3FB950]">The Result</h3>
            </div>
            <p className="text-gray-300 leading-relaxed">
              This two-pass approach ensures that the agent always has the right tools for the job, 
              without being overwhelmed by irrelevant options.
            </p>
          </div>
        </div>

        {/* PHẦN 3: THE API IN ACTION */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] mb-8">
            How to Use It
          </h2>
          
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            Enabling tool retrieval is a simple, three-step process. You define your agent with the full library of tools, 
            create and configure a ToolRetriever instance, and then simply pass it to the Brain.
          </p>

          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8">
            <h4 className="text-xl font-bold text-[#3FB950] mb-6 flex items-center gap-2">
              <span className="text-2xl">💡</span>
              Complete Configuration Example
            </h4>
            <CodeBlock language="python" filename="tool_retrieval_setup.py">
              {toolRetrievalCode}
            </CodeBlock>
          </div>
        </div>

        {/* Benefits Section */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-[#E6EDF3] text-center mb-8">
            Benefits of Intelligent Tool Retrieval
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">🎯</div>
              <h3 className="text-lg font-bold text-[#58A6FF] mb-3 text-center">Focused Execution</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Agent receives only relevant tools, leading to more accurate and efficient task completion.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-[#7C3AED]/10 to-[#6366F1]/10 border border-[#7C3AED]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">📏</div>
              <h3 className="text-lg font-bold text-[#7C3AED] mb-3 text-center">Context Efficiency</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Optimizes context window usage by including only necessary tool descriptions.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">⚡</div>
              <h3 className="text-lg font-bold text-[#3FB950] mb-3 text-center">Faster Training</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Reduces noise in training data, leading to faster convergence and better performance.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-[#F59E0B]/10 to-[#D97706]/10 border border-[#F59E0B]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">🔧</div>
              <h3 className="text-lg font-bold text-[#F59E0B] mb-3 text-center">Scalable Architecture</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Supports hundreds of tools without performance degradation.
              </p>
            </div>
          </div>
        </div>

        {/* CALL TO ACTION */}
        <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12 text-center">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            Ready to Scale Your Tool Library?
          </h2>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Give your agents access to hundreds of tools without the overhead with ToolBrain's intelligent retrieval.
          </p>
          
          <a 
            href="/key-features"
            className="inline-block bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-10 py-4 rounded-lg text-lg font-semibold transition-colors duration-200 shadow-lg hover:shadow-xl"
          >
            Explore More Features
          </a>
        </div>
      </div>
    </Layout>
  );
}