'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function MultipleAlgorithms() {
  const grpoCode = `brain = Brain(
    agent=my_agent,
    reward_func=my_scalar_reward_function,
    learning_algorithm="GRPO",
    num_group_members=8 # GRPO thrives on group comparison
)`;

  const dpoCode = `from toolbrain.rewards import reward_llm_judge_via_ranking

brain = Brain(
    agent=my_agent,
    reward_func=reward_llm_judge_via_ranking,
    learning_algorithm="DPO",
    num_group_members=4 # Needs at least 2 to create pairs
)`;

  const distillationCode = `# The Brain is initialized for a later RL phase (e.g., GRPO)
brain = Brain(agent=student_agent, ..., learning_algorithm="GRPO")

# But first, we run distillation, which uses supervised learning internally
brain.distill(
    dataset=training_tasks,
    teacher_model_id="GPT-4-Turbo"
)

# Now, the agent is pre-trained and ready for RL
brain.train(dataset=training_tasks)`;

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        {/* PHẦN 1: THE POWER OF CHOICE */}
        <div className="text-center mb-20">
          <h1 className="text-5xl font-bold text-[#E6EDF3] mb-8">
            Multiple Learning Algorithms
          </h1>
          
          <div className="max-w-4xl mx-auto space-y-6">
            <p className="text-xl text-gray-400 leading-relaxed">
              Different agentic tasks require different learning strategies. A one-size-fits-all approach is often suboptimal. 
              ToolBrain provides out-of-the-box support for a range of state-of-the-art learning algorithms, allowing you to 
              choose the best approach for your specific problem.
            </p>
            
            <div className="bg-gradient-to-r from-[#58A6FF]/10 to-[#3FB950]/10 border border-[#58A6FF]/30 rounded-xl p-6">
              <p className="text-xl font-semibold text-[#58A6FF]">
                Switching between algorithms is as simple as changing a single string parameter in the Brain constructor.
              </p>
            </div>
          </div>
        </div>

        {/* Algorithm Selection Visual */}
        <div className="mb-20">
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12">
            <h3 className="text-2xl font-bold text-[#E6EDF3] text-center mb-8">
              Choose Your Learning Strategy
            </h3>
            
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-gradient-to-br from-[#58A6FF]/15 to-[#4A90E2]/15 border border-[#58A6FF]/50 rounded-xl p-6 text-center hover:scale-105 transition-transform">
                <div className="text-4xl mb-4">📊</div>
                <h4 className="text-xl font-bold text-[#58A6FF] mb-3">GRPO</h4>
                <p className="text-gray-300 text-sm">Group Relative Policy Optimization</p>
                <p className="text-[#3FB950] text-sm mt-2">Best for scalar rewards</p>
              </div>
              
              <div className="bg-gradient-to-br from-[#7C3AED]/15 to-[#6366F1]/15 border border-[#7C3AED]/50 rounded-xl p-6 text-center hover:scale-105 transition-transform">
                <div className="text-4xl mb-4">⚖️</div>
                <h4 className="text-xl font-bold text-[#7C3AED] mb-3">DPO</h4>
                <p className="text-gray-300 text-sm">Direct Preference Optimization</p>
                <p className="text-[#3FB950] text-sm mt-2">Best for preference feedback</p>
              </div>
              
              <div className="bg-gradient-to-br from-[#3FB950]/15 to-[#10B981]/15 border border-[#3FB950]/50 rounded-xl p-6 text-center hover:scale-105 transition-transform">
                <div className="text-4xl mb-4">🎓</div>
                <h4 className="text-xl font-bold text-[#3FB950] mb-3">Distillation</h4>
                <p className="text-gray-300 text-sm">Supervised Learning</p>
                <p className="text-[#3FB950] text-sm mt-2">Best for knowledge transfer</p>
              </div>
            </div>
          </div>
        </div>

        {/* PHẦN 2: SUPPORTED ALGORITHMS */}
        <div className="space-y-16">
          
          {/* 1. GRPO */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8">
            <div className="grid lg:grid-cols-2 gap-8 items-start">
              <div>
                <div className="flex items-center gap-4 mb-6">
                  <div className="text-4xl">📊</div>
                  <div>
                    <h2 className="text-3xl font-bold text-[#58A6FF] mb-2">
                      GRPO
                    </h2>
                    <p className="text-xl text-gray-300">Group Relative Policy Optimization</p>
                  </div>
                </div>
                
                <div className="mb-6">
                  <div className="bg-[#58A6FF]/10 border border-[#58A6FF]/30 rounded-lg p-4 mb-4">
                    <p className="text-[#58A6FF] font-semibold mb-2">🎯 Best for:</p>
                    <p className="text-gray-300">Scenarios with clear, scalar reward signals.</p>
                  </div>
                  
                  <p className="text-gray-300 leading-relaxed">
                    GRPO is a powerful and sample-efficient policy gradient algorithm. It works by generating a group of attempts (Traces) 
                    for a single query and normalizing their rewards within that group. This creates a stable "relative advantage" signal 
                    that guides the agent's learning, balancing exploration and exploitation effectively. It's an excellent default choice 
                    for many tool-use tasks.
                  </p>
                </div>
              </div>
              
              <div>
                <h4 className="text-lg font-bold text-[#3FB950] mb-4 flex items-center gap-2">
                  <span className="text-xl">💡</span>
                  Usage Example
                </h4>
                <CodeBlock language="python" filename="grpo_example.py">
                  {grpoCode}
                </CodeBlock>
              </div>
            </div>
          </div>

          {/* 2. DPO */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8">
            <div className="grid lg:grid-cols-2 gap-8 items-start">
              <div>
                <div className="flex items-center gap-4 mb-6">
                  <div className="text-4xl">⚖️</div>
                  <div>
                    <h2 className="text-3xl font-bold text-[#7C3AED] mb-2">
                      DPO
                    </h2>
                    <p className="text-xl text-gray-300">Direct Preference Optimization</p>
                  </div>
                </div>
                
                <div className="mb-6">
                  <div className="bg-[#7C3AED]/10 border border-[#7C3AED]/30 rounded-lg p-4 mb-4">
                    <p className="text-[#7C3AED] font-semibold mb-2">🎯 Best for:</p>
                    <p className="text-gray-300">Scenarios with preference-based or comparative feedback (like an LLM-as-a-Judge).</p>
                  </div>
                  
                  <p className="text-gray-300 leading-relaxed">
                    DPO is a simpler and often more stable alternative to traditional RLHF pipelines. It learns directly from preference pairs 
                    (chosen vs. rejected examples). ToolBrain seamlessly integrates DPO with our ranking-based LLM-as-a-Judge, which automatically 
                    generates these preference pairs by ranking multiple agent attempts.
                  </p>
                </div>
              </div>
              
              <div>
                <h4 className="text-lg font-bold text-[#3FB950] mb-4 flex items-center gap-2">
                  <span className="text-xl">💡</span>
                  Usage Example
                </h4>
                <CodeBlock language="python" filename="dpo_example.py">
                  {dpoCode}
                </CodeBlock>
              </div>
            </div>
          </div>

          {/* 3. Supervised Learning / Distillation */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8">
            <div className="grid lg:grid-cols-2 gap-8 items-start">
              <div>
                <div className="flex items-center gap-4 mb-6">
                  <div className="text-4xl">🎓</div>
                  <div>
                    <h2 className="text-3xl font-bold text-[#3FB950] mb-2">
                      Supervised Learning / Distillation
                    </h2>
                    <p className="text-xl text-gray-300">Knowledge Transfer</p>
                  </div>
                </div>
                
                <div className="mb-6">
                  <div className="bg-[#3FB950]/10 border border-[#3FB950]/30 rounded-lg p-4 mb-4">
                    <p className="text-[#3FB950] font-semibold mb-2">🎯 Best for:</p>
                    <p className="text-gray-300">Pre-training, knowledge transfer, or when you have a dataset of "expert" examples.</p>
                  </div>
                  
                  <p className="text-gray-300 leading-relaxed">
                    ToolBrain also supports standard supervised fine-tuning. This is the core mechanism behind our Knowledge Distillation feature, 
                    where the Brain uses a powerful "teacher" model to generate high-quality Execution Traces. The "student" agent then learns from 
                    these expert demonstrations via a supervised loss before starting any RL training.
                  </p>
                </div>
              </div>
              
              <div>
                <h4 className="text-lg font-bold text-[#3FB950] mb-4 flex items-center gap-2">
                  <span className="text-xl">💡</span>
                  Distillation Example
                </h4>
                <CodeBlock language="python" filename="distillation_example.py">
                  {distillationCode}
                </CodeBlock>
              </div>
            </div>
          </div>
        </div>

        {/* Algorithm Comparison Table */}
        <div className="mt-20 mb-16">
          <h2 className="text-3xl font-bold text-[#E6EDF3] text-center mb-8">
            Algorithm Comparison Guide
          </h2>
          
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-[#0D1117] border-b border-[#30363D]">
                    <th className="text-left p-6 text-[#E6EDF3] font-bold">Algorithm</th>
                    <th className="text-left p-6 text-[#E6EDF3] font-bold">Best Use Case</th>
                    <th className="text-left p-6 text-[#E6EDF3] font-bold">Reward Type</th>
                    <th className="text-left p-6 text-[#E6EDF3] font-bold">Sample Efficiency</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-[#30363D]/50 hover:bg-[#58A6FF]/5">
                    <td className="p-6">
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">📊</span>
                        <div>
                          <div className="font-bold text-[#58A6FF]">GRPO</div>
                          <div className="text-sm text-gray-400">Policy Gradient</div>
                        </div>
                      </div>
                    </td>
                    <td className="p-6 text-gray-300">Clear, objective tasks</td>
                    <td className="p-6 text-gray-300">Scalar rewards</td>
                    <td className="p-6">
                      <span className="bg-[#3FB950]/20 text-[#3FB950] px-3 py-1 rounded-full text-sm font-medium">High</span>
                    </td>
                  </tr>
                  <tr className="border-b border-[#30363D]/50 hover:bg-[#7C3AED]/5">
                    <td className="p-6">
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">⚖️</span>
                        <div>
                          <div className="font-bold text-[#7C3AED]">DPO</div>
                          <div className="text-sm text-gray-400">Preference Learning</div>
                        </div>
                      </div>
                    </td>
                    <td className="p-6 text-gray-300">Subjective, semantic tasks</td>
                    <td className="p-6 text-gray-300">Preference pairs</td>
                    <td className="p-6">
                      <span className="bg-[#58A6FF]/20 text-[#58A6FF] px-3 py-1 rounded-full text-sm font-medium">Medium</span>
                    </td>
                  </tr>
                  <tr className="hover:bg-[#3FB950]/5">
                    <td className="p-6">
                      <div className="flex items-center gap-3">
                        <span className="text-2xl">🎓</span>
                        <div>
                          <div className="font-bold text-[#3FB950]">Distillation</div>
                          <div className="text-sm text-gray-400">Supervised Learning</div>
                        </div>
                      </div>
                    </td>
                    <td className="p-6 text-gray-300">Knowledge transfer, pre-training</td>
                    <td className="p-6 text-gray-300">Expert demonstrations</td>
                    <td className="p-6">
                      <span className="bg-[#3FB950]/20 text-[#3FB950] px-3 py-1 rounded-full text-sm font-medium">Very High</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* CALL TO ACTION */}
        <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12 text-center">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            Ready to Experiment with Different Algorithms?
          </h2>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Start with GRPO for most tasks, try DPO for subjective evaluation, or use Distillation to bootstrap your training.
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