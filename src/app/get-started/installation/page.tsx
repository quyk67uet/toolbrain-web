'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function Installation() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-[#E6EDF3] mb-6">Installation</h1>
          <p className="text-xl text-gray-400 leading-relaxed">
            ToolBrain is designed to be lightweight and modular. You can start with a minimal installation 
            and add features as you need them. We recommend using pip and a virtual environment for installation.
          </p>
        </div>

        {/* 1. Standard Installation */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">1. Standard Installation</h2>
          <p className="text-lg text-gray-400 mb-6 leading-relaxed">
            This is the recommended starting point. It installs the core toolbrain library, which is 
            CPU-compatible and allows you to explore the main APIs and run basic examples.
          </p>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
            <CodeBlock language="bash">
              pip install toolbrain
            </CodeBlock>
          </div>
        </div>

        {/* 2. GPU-Accelerated Training */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">2. GPU-Accelerated Training (Recommended for Performance)</h2>
          <div className="mb-6">
            <p className="text-lg text-gray-400 mb-4 leading-relaxed">
              To unlock the full power of ToolBrain, including significant speed improvements and memory 
              savings via Unsloth and QLoRA, we highly recommend installing with the <code className="bg-[#21262D] px-2 py-1 rounded text-[#58A6FF]">[unsloth]</code> extra. 
              This is essential for training larger models or working with long context windows.
            </p>
            <div className="bg-[#2D1B1B] border border-[#F85149]/30 rounded-lg p-4">
              <p className="text-[#F85149] font-medium mb-2">⚠️ Note:</p>
              <p className="text-gray-300">This requires an NVIDIA GPU with CUDA support.</p>
            </div>
          </div>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
            <CodeBlock language="bash">
              pip install &quot;toolbrain[unsloth]&quot;
            </CodeBlock>
          </div>
        </div>

        {/* 3. Installation for Running Examples */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">3. Installation for Running Examples</h2>
          <p className="text-lg text-gray-400 mb-8 leading-relaxed">
            Our repository includes a rich set of examples to help you get started. We offer two options 
            for installing the necessary dependencies.
          </p>

          {/* 3.1 Basic Examples */}
          <div className="mb-8">
            <h3 className="text-2xl font-semibold text-[#E6EDF3] mb-4">3.1. For Basic Examples (CPU/Mac Compatible)</h3>
            <p className="text-lg text-gray-400 mb-6 leading-relaxed">
              To run simple examples and generate plots, install the <code className="bg-[#21262D] px-2 py-1 rounded text-[#58A6FF]">[examples-base]</code> extra. 
              This includes libraries like pandas and matplotlib.
            </p>
            
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <CodeBlock language="bash">
                pip install &quot;toolbrain[examples-base]&quot;
              </CodeBlock>
            </div>
          </div>

          {/* 3.2 All Examples */}
          <div className="mb-8">
            <h3 className="text-2xl font-semibold text-[#E6EDF3] mb-4">3.2. For All Examples (GPU Required)</h3>
            <p className="text-lg text-gray-400 mb-6 leading-relaxed">
              To run all examples, including the advanced, high-performance use cases like the Email Search Agent, 
              install the <code className="bg-[#21262D] px-2 py-1 rounded text-[#58A6FF]">[examples-full]</code> extra. 
              This includes all base example dependencies plus Unsloth for GPU acceleration.
            </p>
            
            <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
              <CodeBlock language="bash">
                pip install &quot;toolbrain[examples-full]&quot;
              </CodeBlock>
            </div>
          </div>
        </div>

        {/* 4. Full Development Setup */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">4. Full Development Setup</h2>
          <p className="text-lg text-gray-400 mb-6 leading-relaxed">
            If you wish to contribute to ToolBrain or install all possible dependencies, you should first 
            clone the repository and then install the library in editable mode with the <code className="bg-[#21262D] px-2 py-1 rounded text-[#58A6FF]">[all]</code> extra.
          </p>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
            <CodeBlock language="bash">
{`# 1. Clone the repository
git clone https://github.com/your-repo/ToolBrain.git
cd ToolBrain

# 2. Install in editable mode with all extras
pip install -e ".[all]"`}
            </CodeBlock>
          </div>
        </div>

        {/* Installation Summary Table */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">Installation Options Summary</h2>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-[#21262D]">
                  <tr>
                    <th className="text-left text-[#E6EDF3] font-semibold p-4">Installation Type</th>
                    <th className="text-left text-[#E6EDF3] font-semibold p-4">Command</th>
                    <th className="text-left text-[#E6EDF3] font-semibold p-4">Use Case</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t border-[#30363D]">
                    <td className="p-4 text-[#E6EDF3] font-medium">Standard</td>
                    <td className="p-4 text-gray-400 font-mono text-sm">pip install toolbrain</td>
                    <td className="p-4 text-gray-400">Basic usage, CPU training</td>
                  </tr>
                  <tr className="border-t border-[#30363D]">
                    <td className="p-4 text-[#E6EDF3] font-medium">GPU-Accelerated</td>
                    <td className="p-4 text-gray-400 font-mono text-sm">pip install &quot;toolbrain[unsloth]&quot;</td>
                    <td className="p-4 text-gray-400">High-performance training</td>
                  </tr>
                  <tr className="border-t border-[#30363D]">
                    <td className="p-4 text-[#E6EDF3] font-medium">Basic Examples</td>
                    <td className="p-4 text-gray-400 font-mono text-sm">pip install &quot;toolbrain[examples-base]&quot;</td>
                    <td className="p-4 text-gray-400">Run simple examples</td>
                  </tr>
                  <tr className="border-t border-[#30363D]">
                    <td className="p-4 text-[#E6EDF3] font-medium">All Examples</td>
                    <td className="p-4 text-gray-400 font-mono text-sm">pip install &quot;toolbrain[examples-full]&quot;</td>
                    <td className="p-4 text-gray-400">Run all examples with GPU</td>
                  </tr>
                  <tr className="border-t border-[#30363D]">
                    <td className="p-4 text-[#E6EDF3] font-medium">Development</td>
                    <td className="p-4 text-gray-400 font-mono text-sm">pip install -e &quot;.[all]&quot;</td>
                    <td className="p-4 text-gray-400">Contributing, full features</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Verification */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">Verify Your Installation</h2>
          <p className="text-lg text-gray-400 mb-6 leading-relaxed">
            After installation, verify that ToolBrain is working correctly:
          </p>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
            <CodeBlock language="python">
{`# Quick verification test
import toolbrain
print(f"ToolBrain version: {toolbrain.__version__}")

# Test basic functionality
from toolbrain import Brain
brain = Brain()
print("✅ ToolBrain installed and working correctly!")`}
            </CodeBlock>
          </div>
        </div>

        {/* Next Steps */}
        <div className="bg-gradient-to-r from-[#58A6FF]/10 to-[#7C3AED]/10 border border-[#58A6FF]/20 rounded-lg p-8 text-center">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-4">🚀 Ready for the Next Step?</h2>
          <p className="text-gray-300 mb-6">
            Perfect! ToolBrain is now installed. Let&apos;s train your first agent in the quickstart guide.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a 
              href="/get-started/quickstart"
              className="bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-6 py-3 rounded-lg font-medium transition-colors duration-200"
            >
              → Start Quickstart Guide
            </a>
            <a 
              href="/tutorials"
              className="border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] px-6 py-3 rounded-lg font-medium transition-colors duration-200"
            >
              Browse Tutorials
            </a>
          </div>
        </div>
      </div>
    </Layout>
  );
}