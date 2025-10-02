'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function KnowledgeDistillation() {
  const distillationCode = `from toolbrain import Brain
from smolagents import CodeAgent
from toolbrain.models import UnslothModel

# 1. Create your small "student" agent
student_agent = CodeAgent(
    model=UnslothModel(model_id="Qwen/Qwen2.5-0.5B-Instruct"),
    tools=[...]
)

# 2. Initialize the Brain with the student
brain = Brain(
    agent=student_agent,
    reward_func=my_reward_function,
    learning_algorithm="GRPO"
)

# 3. Run distillation as a pre-training step
# The Brain handles the teacher model internally.
print("--- Phase 1: Knowledge Distillation ---")
brain.distill(
    dataset=training_dataset,
    teacher_model_id="Qwen/Qwen2.5-7B-Instruct" # Specify the teacher
)

# 4. Continue with standard RL training on the now "warmed-up" student
print("\\n--- Phase 2: Reinforcement Learning ---")
brain.train(dataset=training_dataset, num_iterations=10)`;

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        {/* PHẦN 1: THE PROBLEM - "MODEL NHỎ KHÓ HỌC" */}
        <div className="text-center mb-20">
          <h1 className="text-5xl font-bold text-[#E6EDF3] mb-8">
            Knowledge Distillation
          </h1>
          
          <div className="max-w-4xl mx-auto space-y-6">
            <p className="text-xl text-gray-400 leading-relaxed">
              When training agents with smaller, more efficient language models, you may observe slow convergence and 
              poor performance during the initial stages of Reinforcement Learning. This is because small models have 
              a limited capacity and struggle with the inefficient exploration required at the beginning of RL training.
            </p>
            
            <div className="bg-gradient-to-r from-[#58A6FF]/10 to-[#3FB950]/10 border border-[#58A6FF]/30 rounded-xl p-8">
              <div className="flex items-center justify-center gap-4 mb-4">
                <div className="text-3xl">💡</div>
                <h3 className="text-2xl font-bold text-[#58A6FF]">The Solution</h3>
              </div>
              <p className="text-xl font-semibold text-[#E6EDF3]">
                How can we give our small "student" model a massive head start? 
                By letting it learn from a powerful "teacher" model first.
              </p>
            </div>
          </div>
        </div>

        {/* Benefits Comparison */}
        <div className="mb-20">
          <h2 className="text-3xl font-bold text-[#E6EDF3] text-center mb-8">
            The Power of Starting with Expert Knowledge
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-xl p-8">
              <div className="text-center">
                <div className="text-4xl mb-4">🏋️</div>
                <h3 className="text-xl font-bold text-[#58A6FF] mb-4">Traditional Training</h3>
                <p className="text-gray-400 text-sm mb-4">Starting from scratch</p>
                <div className="space-y-3 text-left">
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#58A6FF] rounded-full"></span>
                    Models learn through trial and error
                  </p>
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#58A6FF] rounded-full"></span>
                    Extended exploration phase required
                  </p>
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#58A6FF] rounded-full"></span>
                    More training iterations needed
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-xl p-8">
              <div className="text-center">
                <div className="text-4xl mb-4">🚀</div>
                <h3 className="text-xl font-bold text-[#3FB950] mb-4">With Knowledge Distillation</h3>
                <p className="text-gray-400 text-sm mb-4">Starting with expert guidance</p>
                <div className="space-y-3 text-left">
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#3FB950] rounded-full"></span>
                    Pre-trained with expert demonstrations
                  </p>
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#3FB950] rounded-full"></span>
                    Accelerated learning from day one
                  </p>
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#3FB950] rounded-full"></span>
                    Superior final performance achieved
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* PHẦN 2: THE SOLUTION - DISTILLATION PIPELINE */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] text-center mb-8">
            The Distillation Pipeline in ToolBrain
          </h2>
          
          <p className="text-lg text-gray-400 mb-12 text-center max-w-3xl mx-auto leading-relaxed">
            ToolBrain provides a simple <code className="bg-[#161B22] px-2 py-1 rounded text-[#58A6FF]">.distill()</code> method 
            that automates the entire knowledge distillation process. Here's what happens under the hood:
          </p>

          {/* VISUAL PIPELINE */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12">
            
            {/* Mobile Layout */}
            <div className="flex flex-col lg:hidden space-y-12">
              
              {/* Step 1: Generate */}
              <div className="text-center">
                <div className="bg-gradient-to-br from-[#58A6FF]/20 to-[#4A90E2]/20 border-2 border-[#58A6FF] rounded-xl p-8 mb-6">
                  <div className="flex items-center justify-center gap-4 mb-4">
                    <div className="text-5xl">👨‍🏫</div>
                    <div className="text-3xl">→</div>
                    <div className="flex gap-2">
                      <div className="text-2xl">📜</div>
                      <div className="text-2xl">📜</div>
                      <div className="text-2xl">📜</div>
                    </div>
                  </div>
                  <h3 className="text-2xl font-bold text-[#58A6FF] mb-3">1. Generate</h3>
                  <p className="text-gray-300 leading-relaxed">
                    <span className="font-semibold text-[#58A6FF]">Teacher Generation:</span> A large, powerful "teacher" model 
                    (e.g., GPT-4) runs the tasks and generates high-quality, successful Execution Traces.
                  </p>
                </div>
              </div>

              {/* Step 2: Filter */}
              <div className="text-center">
                <div className="bg-gradient-to-br from-[#7C3AED]/20 to-[#6366F1]/20 border-2 border-[#7C3AED] rounded-xl p-8 mb-6">
                  <div className="flex items-center justify-center gap-4 mb-4">
                    <div className="text-5xl">🔽</div>
                    <div className="text-3xl">→</div>
                    <div className="flex gap-2">
                      <div className="text-2xl">✨</div>
                      <div className="text-2xl">🏆</div>
                    </div>
                  </div>
                  <h3 className="text-2xl font-bold text-[#7C3AED] mb-3">2. Filter</h3>
                  <p className="text-gray-300 leading-relaxed">
                    <span className="font-semibold text-[#7C3AED]">Quality Filtering:</span> ToolBrain automatically filters 
                    these traces, keeping only the most successful examples.
                  </p>
                </div>
              </div>

              {/* Step 3: Train */}
              <div className="text-center">
                <div className="bg-gradient-to-br from-[#3FB950]/20 to-[#10B981]/20 border-2 border-[#3FB950] rounded-xl p-8">
                  <div className="flex items-center justify-center gap-4 mb-4">
                    <div className="flex gap-2">
                      <div className="text-2xl">✨</div>
                      <div className="text-2xl">🏆</div>
                    </div>
                    <div className="text-3xl">→</div>
                    <div className="text-5xl">👨‍🎓</div>
                  </div>
                  <h3 className="text-2xl font-bold text-[#3FB950] mb-3">3. Train</h3>
                  <p className="text-gray-300 leading-relaxed">
                    <span className="font-semibold text-[#3FB950]">Supervised Learning:</span> The small "student" model is 
                    then trained on this curated dataset of expert demonstrations using a standard supervised learning loss.
                  </p>
                </div>
              </div>
            </div>

            {/* Desktop Layout */}
            <div className="hidden lg:flex items-center justify-center gap-8">
              
              {/* Step 1: Generate */}
              <div className="flex-1">
                <div className="bg-gradient-to-br from-[#58A6FF]/20 to-[#4A90E2]/20 border-2 border-[#58A6FF] rounded-xl p-6 text-center">
                  <div className="text-5xl mb-4">👨‍🏫</div>
                  <h3 className="text-xl font-bold text-[#58A6FF] mb-3">1. Generate</h3>
                  <div className="flex justify-center gap-1 mb-4">
                    <div className="text-xl">📜</div>
                    <div className="text-xl">📜</div>
                    <div className="text-xl">📜</div>
                  </div>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    Teacher model generates high-quality Execution Traces
                  </p>
                </div>
              </div>

              {/* Arrow 1 */}
              <div className="flex justify-center">
                <svg className="w-12 h-8 text-[#58A6FF]" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M13.025 1l-2.847 2.828 6.176 6.176h-16.354v3.992h16.354l-6.176 6.176 2.847 2.828 10.975-11z"/>
                </svg>
              </div>

              {/* Step 2: Filter */}
              <div className="flex-1">
                <div className="bg-gradient-to-br from-[#7C3AED]/20 to-[#6366F1]/20 border-2 border-[#7C3AED] rounded-xl p-6 text-center">
                  <div className="text-5xl mb-4">🔽</div>
                  <h3 className="text-xl font-bold text-[#7C3AED] mb-3">2. Filter</h3>
                  <div className="flex justify-center gap-1 mb-4">
                    <div className="text-xl">✨</div>
                    <div className="text-xl">🏆</div>
                  </div>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    Automatic quality filtering keeps only the best examples
                  </p>
                </div>
              </div>

              {/* Arrow 2 */}
              <div className="flex justify-center">
                <svg className="w-12 h-8 text-[#58A6FF]" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M13.025 1l-2.847 2.828 6.176 6.176h-16.354v3.992h16.354l-6.176 6.176 2.847 2.828 10.975-11z"/>
                </svg>
              </div>

              {/* Step 3: Train */}
              <div className="flex-1">
                <div className="bg-gradient-to-br from-[#3FB950]/20 to-[#10B981]/20 border-2 border-[#3FB950] rounded-xl p-6 text-center">
                  <div className="text-5xl mb-4">👨‍🎓</div>
                  <h3 className="text-xl font-bold text-[#3FB950] mb-3">3. Train</h3>
                  <div className="flex justify-center gap-1 mb-4">
                    <div className="text-xl">🧠</div>
                    <div className="text-xl">📚</div>
                  </div>
                  <p className="text-sm text-gray-300 leading-relaxed">
                    Student model learns from expert demonstrations
                  </p>
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
              The result is a student model that has been "warmed up" with expert knowledge, making it ready for 
              much more efficient and effective Reinforcement Learning fine-tuning.
            </p>
          </div>
        </div>

        {/* PHẦN 3: THE API IN ACTION */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] mb-8">
            How to Use It
          </h2>
          
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            Integrating distillation into your workflow is incredibly simple. You first initialize your Brain with your 
            small student agent, and then call the <code className="bg-[#161B22] px-2 py-1 rounded text-[#58A6FF]">.distill()</code> method 
            before your main <code className="bg-[#161B22] px-2 py-1 rounded text-[#58A6FF]">.train()</code> call.
          </p>

          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8">
            <h4 className="text-xl font-bold text-[#3FB950] mb-6 flex items-center gap-2">
              <span className="text-2xl">💡</span>
              Complete Example
            </h4>
            <CodeBlock language="python" filename="distillation_workflow.py">
              {distillationCode}
            </CodeBlock>
          </div>
        </div>

        {/* Benefits Section */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-[#E6EDF3] text-center mb-8">
            Why Knowledge Distillation Works
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">⚡</div>
              <h3 className="text-xl font-bold text-[#58A6FF] mb-3 text-center">Faster Convergence</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Student models start with expert knowledge, dramatically reducing the time to reach good performance.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-[#7C3AED]/10 to-[#6366F1]/10 border border-[#7C3AED]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">🎯</div>
              <h3 className="text-xl font-bold text-[#7C3AED] mb-3 text-center">Better Final Performance</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Starting from a good baseline often leads to better final performance compared to random initialization.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">💰</div>
              <h3 className="text-xl font-bold text-[#3FB950] mb-3 text-center">Cost Efficient</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Reduces the computational cost of training by requiring fewer RL iterations to reach target performance.
              </p>
            </div>
          </div>
        </div>

        {/* CALL TO ACTION */}
        <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12 text-center">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            Ready to Accelerate Your Training?
          </h2>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Give your small models a massive head start with ToolBrain's knowledge distillation.
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