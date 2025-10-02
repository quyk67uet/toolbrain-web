'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import Image from 'next/image';

export default function FlexibleRewards() {
  const rewardEfficiencyCode = `from toolbrain.core_types import Trace

def reward_step_efficiency(trace: Trace, **kwargs) -> float:
    """Rewards higher for shorter traces."""
    num_turns = len(trace)
    if num_turns <= 3: # Ideal number of turns
        return 1.0
    else:
        # Penalize for each step over the ideal
        penalty = (num_turns - 3) * 0.2
        return max(0.0, 1.0 - penalty)`;

  const userDefinedUsage = `# Simply pass your function to the Brain
brain = Brain(
    agent=my_agent,
    reward_func=reward_step_efficiency,
    ...
)`;

  const llmJudgeUsage = `from toolbrain.rewards import reward_llm_judge_via_ranking

# Simply assign the built-in judge function
brain = Brain(
    agent=my_agent,
    reward_func=reward_llm_judge_via_ranking,
    ...
)`;

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        {/* PHẦN 1: THE CHALLENGE OF REWARD DESIGN */}
        <div className="text-center mb-20">
          <h1 className="text-5xl font-bold text-[#E6EDF3] mb-8">
            Flexible, Hybrid Rewards
          </h1>
          
          <div className="max-w-4xl mx-auto text-lg text-gray-400 leading-relaxed space-y-6">
            <p>
              The reward function is the single most important component in a Reinforcement Learning system—it defines the goal. 
              However, designing a good reward function is notoriously difficult.
            </p>
            
            <div className="grid md:grid-cols-2 gap-8 my-12">
              <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-xl p-6">
                <div className="text-3xl mb-4">📏</div>
                <h3 className="text-xl font-bold text-[#58A6FF] mb-3">User-Defined Rewards</h3>
                <div className="space-y-2">
                  <p className="text-[#3FB950] text-sm leading-relaxed flex items-center gap-2">
                    <span className="text-[#3FB950]">✓</span> Precise and deterministic control
                  </p>
                  <p className="text-[#3FB950] text-sm leading-relaxed flex items-center gap-2">
                    <span className="text-[#3FB950]">✓</span> Fast execution and low latency
                  </p>
                  <p className="text-[#3FB950] text-sm leading-relaxed flex items-center gap-2">
                    <span className="text-[#3FB950]">✓</span> Perfect for objective criteria
                  </p>
                </div>
              </div>
              
              <div className="bg-gradient-to-br from-[#7C3AED]/10 to-[#6366F1]/10 border border-[#7C3AED]/30 rounded-xl p-6">
                <div className="text-3xl mb-4">🤖</div>
                <h3 className="text-xl font-bold text-[#7C3AED] mb-3">LLM-as-a-Judge Rewards</h3>
                <div className="space-y-2">
                  <p className="text-[#3FB950] text-sm leading-relaxed flex items-center gap-2">
                    <span className="text-[#3FB950]">✓</span> Handles complex semantic tasks
                  </p>
                  <p className="text-[#3FB950] text-sm leading-relaxed flex items-center gap-2">
                    <span className="text-[#3FB950]">✓</span> Understands nuanced quality
                  </p>
                  <p className="text-[#3FB950] text-sm leading-relaxed flex items-center gap-2">
                    <span className="text-[#3FB950]">✓</span> Scales with task complexity
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-xl p-6">
              <div className="flex items-center justify-center gap-3 mb-4">
                <div className="text-2xl">🚀</div>
                <h3 className="text-xl font-bold text-[#3FB950]">ToolBrain's Hybrid Approach</h3>
              </div>
              <p className="text-xl font-semibold text-center text-[#E6EDF3]">
                Combines the precision of user-defined functions with the intelligence of LLM judges, 
                automatically adapting to your specific needs for optimal performance.
              </p>
            </div>
          </div>
        </div>

        {/* PHẦN 2: PATH 1 - USER-DEFINED REWARDS */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] mb-8">
            Path 1: User-Defined Rewards for Precise Control
          </h2>
          
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            For tasks with clear, objective success criteria, nothing beats a simple Python function. 
            ToolBrain allows you to provide any Python callable as a reward function. It will automatically 
            receive the agent's Execution Trace, giving you full access to the agent's thoughts, actions, 
            and results to implement your custom logic.
          </p>

          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8 mb-8">
            <p className="text-lg text-[#58A6FF] mb-6 font-semibold">
              For example, here is a built-in reward function that encourages the agent to be more efficient:
            </p>
            
            <CodeBlock language="python" filename="reward_efficiency.py">
              {rewardEfficiencyCode}
            </CodeBlock>
          </div>

          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
            <h4 className="text-xl font-bold text-[#3FB950] mb-4 flex items-center gap-2">
              <span className="text-2xl">💡</span>
              Usage
            </h4>
            <CodeBlock language="python" filename="usage_example.py">
              {userDefinedUsage}
            </CodeBlock>
          </div>
        </div>

        {/* PHẦN 3: PATH 2 - LLM-AS-A-JUDGE */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] mb-8">
            Path 2: LLM-as-a-Judge for Semantic Tasks
          </h2>
          
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            What if the goal is subjective, like "is this summary good?" For these tasks, ToolBrain provides 
            a powerful, built-in ranking-based LLM-as-a-Judge.
          </p>
          
          <p className="text-lg text-gray-400 mb-12 leading-relaxed">
            Instead of asking an LLM for an unreliable absolute score, our judge shows the LLM multiple agent 
            attempts (Traces) and asks it to rank them from best to worst. This relative comparison is a much 
            easier and more consistent task for LLMs, resulting in a more reliable reward signal.
          </p>

          {/* VISUAL DIAGRAM */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12 mb-8">
            <h3 className="text-2xl font-bold text-[#E6EDF3] text-center mb-8">
              LLM-as-a-Judge Ranking Process
            </h3>
            
            <div className="flex flex-col lg:flex-row items-center justify-center gap-8 lg:gap-12">
              {/* Step 1: Multiple Traces */}
              <div className="flex flex-col items-center">
                <h4 className="text-lg font-semibold text-[#58A6FF] mb-4">Multiple Agent Attempts</h4>
                <div className="flex flex-col space-y-3">
                  <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4 hover:border-[#58A6FF]/50 transition-colors">
                    <div className="flex items-center gap-3">
                      <div className="text-2xl">📝</div>
                      <span className="text-[#E6EDF3] font-medium">Trace A</span>
                    </div>
                  </div>
                  <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4 hover:border-[#58A6FF]/50 transition-colors">
                    <div className="flex items-center gap-3">
                      <div className="text-2xl">📝</div>
                      <span className="text-[#E6EDF3] font-medium">Trace B</span>
                    </div>
                  </div>
                  <div className="bg-[#0D1117] border border-[#30363D] rounded-lg p-4 hover:border-[#58A6FF]/50 transition-colors">
                    <div className="flex items-center gap-3">
                      <div className="text-2xl">📝</div>
                      <span className="text-[#E6EDF3] font-medium">Trace C</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Arrow 1 */}
              <div className="flex justify-center">
                <div className="hidden lg:block">
                  <svg className="w-12 h-8 text-[#58A6FF]" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M13.025 1l-2.847 2.828 6.176 6.176h-16.354v3.992h16.354l-6.176 6.176 2.847 2.828 10.975-11z"/>
                  </svg>
                </div>
                <div className="lg:hidden">
                  <svg className="w-8 h-12 text-[#58A6FF]" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M1 13.025l2.828-2.847 6.176 6.176v-16.354h3.992v16.354l6.176-6.176 2.828 2.847-11 10.975z"/>
                  </svg>
                </div>
              </div>

              {/* Step 2: LLM Judge */}
              <div className="flex flex-col items-center">
                <h4 className="text-lg font-semibold text-[#7C3AED] mb-4">LLM Judge</h4>
                <div className="bg-gradient-to-br from-[#7C3AED]/20 to-[#58A6FF]/20 border-2 border-[#7C3AED] rounded-xl p-6">
                  <div className="text-center">
                    <div className="text-4xl mb-3">⚖️</div>
                    <h4 className="text-xl font-bold text-[#7C3AED]">LLM Judge</h4>
                    <p className="text-sm text-gray-300 mt-2">Comparative<br/>Ranking</p>
                  </div>
                </div>
              </div>

              {/* Arrow 2 */}
              <div className="flex justify-center">
                <div className="hidden lg:block">
                  <svg className="w-12 h-8 text-[#58A6FF]" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M13.025 1l-2.847 2.828 6.176 6.176h-16.354v3.992h16.354l-6.176 6.176 2.847 2.828 10.975-11z"/>
                  </svg>
                </div>
                <div className="lg:hidden">
                  <svg className="w-8 h-12 text-[#58A6FF]" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M1 13.025l2.828-2.847 6.176 6.176v-16.354h3.992v16.354l6.176-6.176 2.828 2.847-11 10.975z"/>
                  </svg>
                </div>
              </div>

              {/* Step 3: Ranking */}
              <div className="flex flex-col items-center">
                <h4 className="text-lg font-semibold text-[#3FB950] mb-4">Ranking Results</h4>
                <div className="flex flex-col space-y-3">
                  <div className="bg-gradient-to-r from-[#FFD700]/20 to-[#FFA500]/20 border border-[#FFD700] rounded-lg p-4">
                    <div className="flex items-center gap-3">
                      <div className="text-2xl">🥇</div>
                      <span className="text-[#E6EDF3] font-medium">Trace B</span>
                      <span className="text-[#3FB950] font-bold">1.0</span>
                    </div>
                  </div>
                  <div className="bg-gradient-to-r from-[#C0C0C0]/20 to-[#A0A0A0]/20 border border-[#C0C0C0] rounded-lg p-4">
                    <div className="flex items-center gap-3">
                      <div className="text-2xl">🥈</div>
                      <span className="text-[#E6EDF3] font-medium">Trace A</span>
                      <span className="text-[#58A6FF] font-bold">0.5</span>
                    </div>
                  </div>
                  <div className="bg-gradient-to-r from-[#CD7F32]/20 to-[#B87333]/20 border border-[#CD7F32] rounded-lg p-4">
                    <div className="flex items-center gap-3">
                      <div className="text-2xl">🥉</div>
                      <span className="text-[#E6EDF3] font-medium">Trace C</span>
                      <span className="text-[#F85149] font-bold">0.0</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
            <h4 className="text-xl font-bold text-[#3FB950] mb-4 flex items-center gap-2">
              <span className="text-2xl">💡</span>
              Usage
            </h4>
            <CodeBlock language="python" filename="llm_judge_usage.py">
              {llmJudgeUsage}
            </CodeBlock>
          </div>
        </div>

        {/* PHẦN 4: THE HYBRID POWER */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] mb-8">
            The Hybrid Power: A Unified Interface
          </h2>
          
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            You might wonder how the Brain can seamlessly handle both a simple function that processes one trace 
            and a complex judge that processes a batch of traces.
          </p>
          
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8">
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div>
                <h3 className="text-2xl font-bold text-[#58A6FF] mb-4">
                  RewardFunctionWrapper
                </h3>
                <p className="text-gray-300 leading-relaxed mb-6">
                  This is handled by an internal component called the <span className="text-[#58A6FF] font-semibold">RewardFunctionWrapper</span>. 
                  It automatically inspects your provided function and adapts the calling strategy accordingly.
                </p>
                <p className="text-gray-300 leading-relaxed">
                  This means you can focus solely on your reward logic, and ToolBrain ensures it integrates 
                  perfectly into the training loop. You can even create your own complex, batch-aware reward functions.
                </p>
              </div>
              
              <div className="relative">
                <div className="bg-gradient-to-br from-[#58A6FF]/20 to-[#7C3AED]/20 border-2 border-[#58A6FF] rounded-xl p-6 text-center">
                  <div className="text-4xl mb-4">🔄</div>
                  <h4 className="text-xl font-bold text-[#58A6FF] mb-2">Automatic</h4>
                  <h4 className="text-xl font-bold text-[#7C3AED] mb-4">Adaptation</h4>
                  <div className="space-y-2 text-sm text-gray-300">
                    <div className="flex items-center justify-center gap-2">
                      <span className="w-2 h-2 bg-[#3FB950] rounded-full"></span>
                      <span>Single trace functions</span>
                    </div>
                    <div className="flex items-center justify-center gap-2">
                      <span className="w-2 h-2 bg-[#58A6FF] rounded-full"></span>
                      <span>Batch processing functions</span>
                    </div>
                    <div className="flex items-center justify-center gap-2">
                      <span className="w-2 h-2 bg-[#7C3AED] rounded-full"></span>
                      <span>Custom reward logic</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* CALL TO ACTION */}
        <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12 text-center">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            Ready to Design Your Perfect Reward System?
          </h2>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Whether you need precise control or semantic understanding, ToolBrain's flexible reward system has you covered.
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