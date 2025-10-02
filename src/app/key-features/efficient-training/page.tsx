'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function EfficientTrainingUnsloth() {
  return (
    <Layout>
      <div className="max-w-6xl mx-auto px-4 py-16">
        
        {/* PHẦN 1: THE PROBLEM - "THE GPU BOTTLENECK" */}
        <div className="mb-20">
          <h1 className="text-5xl font-bold text-[#E6EDF3] mb-8 text-center">
            Efficient & Accessible Training
          </h1>
          
          <div className="max-w-4xl mx-auto text-center mb-12">
            <p className="text-xl text-gray-400 mb-6 leading-relaxed">
              Reinforcement learning for LLM agents is a computationally intensive process. Fine-tuning the underlying models, 
              especially large ones, requires significant GPU memory and can lead to long training cycles. This 
              <span className="text-[#F85149] font-semibold"> "GPU bottleneck"</span> often makes advanced RL training 
              impractical for many developers and researchers.
            </p>
            <p className="text-xl text-gray-300 leading-relaxed">
              ToolBrain is designed from the ground up to be <span className="text-[#3FB950] font-semibold">efficient and accessible</span>. 
              We integrate state-of-the-art optimization techniques directly into our workflow, allowing you to train powerful 
              agents on standard, consumer-grade hardware.
            </p>
          </div>

          {/* GPU Bottleneck Illustration */}
          <div className="grid md:grid-cols-2 gap-8">
            <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-xl p-8">
              <div className="text-center">
                <div className="text-5xl mb-4">🔥</div>
                <h3 className="text-xl font-bold text-[#58A6FF] mb-4">The GPU Challenge</h3>
                <div className="space-y-3 text-left">
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#58A6FF] rounded-full"></span>
                    High memory requirements for model fine-tuning
                  </p>
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#58A6FF] rounded-full"></span>
                    Extended training cycles on standard hardware
                  </p>
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#58A6FF] rounded-full"></span>
                    Limited accessibility for smaller teams
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-xl p-8">
              <div className="text-center">
                <div className="text-5xl mb-4">🚀</div>
                <h3 className="text-xl font-bold text-[#3FB950] mb-4">ToolBrain's Solution</h3>
                <div className="space-y-3 text-left">
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#3FB950] rounded-full"></span>
                    Multi-layered optimization stack built-in
                  </p>
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#3FB950] rounded-full"></span>
                    Consumer-grade hardware compatible
                  </p>
                  <p className="text-gray-300 text-sm flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#3FB950] rounded-full"></span>
                    Accessible to developers and researchers
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* PHẦN 2: THE SOLUTION - A MULTI-LAYERED OPTIMIZATION STACK */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] text-center mb-8">
            Our Optimization Stack
          </h2>
          
          <p className="text-lg text-gray-400 mb-12 text-center max-w-3xl mx-auto leading-relaxed">
            ToolBrain employs a multi-layered stack of optimizations that work together to dramatically reduce 
            memory usage and increase training speed.
          </p>

          {/* OPTIMIZATION STACK VISUAL */}
          <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12">
            <div className="max-w-4xl mx-auto">
              
              {/* Stack Visualization */}
              <div className="flex flex-col space-y-1">
                
                {/* Tầng 3: Unsloth (Top) */}
                <div className="bg-gradient-to-r from-[#FFD700]/20 to-[#FFA500]/20 border border-[#FFD700]/50 rounded-xl p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="text-4xl">⚡</div>
                      <div>
                        <h3 className="text-xl font-bold text-[#FFD700]">Unsloth</h3>
                        <p className="text-sm text-gray-400">Optimized Kernels</p>
                      </div>
                    </div>
                    <div className="text-right max-w-md">
                      <p className="text-sm text-gray-300">
                        We leverage Unsloth's highly optimized CUDA kernels for a 
                        <span className="text-[#FFD700] font-semibold"> 2x speedup</span> in training 
                        and even lower memory usage.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Tầng 2: LoRA/QLoRA (Middle) */}
                <div className="bg-gradient-to-r from-[#7C3AED]/20 to-[#6366F1]/20 border border-[#7C3AED]/50 rounded-xl p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="text-4xl">➕</div>
                      <div>
                        <h3 className="text-xl font-bold text-[#7C3AED]">LoRA / QLoRA</h3>
                        <p className="text-sm text-gray-400">Parameter-Efficient Fine-Tuning</p>
                      </div>
                    </div>
                    <div className="text-right max-w-md">
                      <p className="text-sm text-gray-300">
                        We only train a small number of 
                        <span className="text-[#7C3AED] font-semibold"> "adapter" weights</span> (LoRA), 
                        not the entire model, further saving memory and compute.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Tầng 1: BitsAndBytes (Bottom) */}
                <div className="bg-gradient-to-r from-[#58A6FF]/20 to-[#4A90E2]/20 border border-[#58A6FF]/50 rounded-xl p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="text-4xl">🔢</div>
                      <div>
                        <h3 className="text-xl font-bold text-[#58A6FF]">BitsAndBytes</h3>
                        <p className="text-sm text-gray-400">4-bit Quantization</p>
                      </div>
                    </div>
                    <div className="text-right max-w-md">
                      <p className="text-sm text-gray-300">
                        Models are loaded in <span className="text-[#58A6FF] font-semibold">4-bit precision</span>, 
                        drastically reducing their memory footprint in the GPU.
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-[#3FB950] mb-1">75%</div>
                  <p className="text-xs text-gray-400">Memory Reduction</p>
                </div>
                <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-[#FFD700] mb-1">2x</div>
                  <p className="text-xs text-gray-400">Training Speed</p>
                </div>
                <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-[#58A6FF] mb-1">4-bit</div>
                  <p className="text-xs text-gray-400">Precision</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* PHẦN 3: THE API IN ACTION - AUTOMATIC & SIMPLE */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] text-center mb-8">
            How to Use It: Optimization by Default
          </h2>
          
          <p className="text-lg text-gray-400 mb-12 text-center max-w-3xl mx-auto leading-relaxed">
            The best part? Most of these optimizations are enabled by default when you use our custom model classes. 
            ToolBrain is designed to be efficient out-of-the-box.
          </p>

          {/* Code Example 1: Using UnslothModel */}
          <div className="mb-12">
            <h3 className="text-2xl font-bold text-[#FFD700] mb-4">Using UnslothModel</h3>
            <p className="text-gray-400 mb-4">
              Our custom <code className="bg-[#161B22] px-2 py-1 rounded text-[#FFD700]">UnslothModel</code> class 
              automatically handles 4-bit quantization and prepares the model for Unsloth's speed optimizations. 
              You simply use it in place of a standard model.
            </p>
            <CodeBlock language="python">
{`from toolbrain.models import UnslothModel
from smolagents import CodeAgent

# This single line enables BitsAndBytes, QLoRA, and Unsloth optimizations.
optimized_model = UnslothModel(model_id="Qwen/Qwen2.5-7B-Instruct")

# Your agent is now ready for efficient training.
agent = CodeAgent(
    model=optimized_model,
    tools=[...]
)`}
            </CodeBlock>
          </div>

          {/* Code Example 2: Configuring LoRA */}
          <div className="mb-12">
            <h3 className="text-2xl font-bold text-[#7C3AED] mb-4">Configuring LoRA</h3>
            <p className="text-gray-400 mb-4">
              You can still have granular control. The LoRA configuration can be easily customized and passed directly to the Brain.
            </p>
            <CodeBlock language="python">
{`from peft import LoraConfig
from toolbrain import Brain

# Define a custom LoRA configuration
my_lora_config = LoraConfig(
    r=16, # Increase rank for more capacity
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)

# Pass the config to the Brain
brain = Brain(
    agent=my_agent,
    reward_func=...,
    lora_config=my_lora_config # Override the default LoRA settings
)`}
            </CodeBlock>
          </div>
        </div>

        {/* Benefits Summary */}
        <div className="mb-20">
          <h2 className="text-3xl font-bold text-[#E6EDF3] text-center mb-8">
            Why Efficient Training Matters
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-gradient-to-br from-[#58A6FF]/10 to-[#4A90E2]/10 border border-[#58A6FF]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">💰</div>
              <h3 className="text-xl font-bold text-[#58A6FF] mb-3 text-center">Cost Effective</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Train on consumer hardware instead of expensive cloud GPUs. Reduce compute costs by up to 75%.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-[#3FB950]/10 to-[#10B981]/10 border border-[#3FB950]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">🌍</div>
              <h3 className="text-xl font-bold text-[#3FB950] mb-3 text-center">Democratized Access</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                Enable researchers and small teams to experiment with advanced RL techniques without barriers.
              </p>
            </div>
            
            <div className="bg-gradient-to-br from-[#FFD700]/10 to-[#FFA500]/10 border border-[#FFD700]/30 rounded-xl p-6">
              <div className="text-3xl mb-4 text-center">⚡</div>
              <h3 className="text-xl font-bold text-[#FFD700] mb-3 text-center">Faster Iteration</h3>
              <p className="text-gray-300 text-sm text-center leading-relaxed">
                2x training speed means faster experimentation cycles and quicker time to production.
              </p>
            </div>
          </div>
        </div>

        {/* CALL TO ACTION */}
        <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12 text-center">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            Ready to Train Efficiently?
          </h2>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Experience the power of optimized RL training with ToolBrain's built-in efficiency stack.
          </p>
          
          <a 
            href="/get-started/quickstart"
            className="inline-block bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-10 py-4 rounded-lg text-lg font-semibold transition-colors duration-200 shadow-lg hover:shadow-xl"
          >
            Start Training Now
          </a>
        </div>

      </div>
    </Layout>
  );
}