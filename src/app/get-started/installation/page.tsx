'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function Installation() {
  const systemRequirements = [
    { name: 'Python', version: '3.8+', status: 'required' },
    { name: 'pip', version: 'Latest', status: 'required' },
    { name: 'CUDA', version: '11.0+', status: 'optional' },
    { name: 'Git', version: '2.0+', status: 'recommended' }
  ];

  const installationSteps = [
    {
      title: 'System Preparation',
      description: 'Ensure your system meets the minimum requirements',
      code: `# Check Python version
python --version

# Check pip version  
pip --version

# Update pip (recommended)
python -m pip install --upgrade pip`
    },
    {
      title: 'Virtual Environment Setup',
      description: 'Create isolated environment for ToolBrain',
      code: `# Create virtual environment
python -m venv toolbrain-env

# Activate virtual environment
# Windows:
toolbrain-env\\Scripts\\activate
# Linux/Mac:
source toolbrain-env/bin/activate`
    },
    {
      title: 'Install ToolBrain',
      description: 'Install the main package and dependencies',
      code: `# Install from PyPI (recommended)
pip install toolbrain

# Or install with extra dependencies
pip install toolbrain[full]

# For development version
pip install git+https://github.com/toolbrain/toolbrain.git`
    },
    {
      title: 'GPU Support (Optional)',
      description: 'Install CUDA support for faster training',
      code: `# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')""`
    }
  ];

  const verificationTests = [
    {
      title: 'Basic Import Test',
      code: `import toolbrain
print(f"ToolBrain version: {toolbrain.__version__}")
print("✅ ToolBrain imported successfully!")`
    },
    {
      title: 'Core Components Test',
      code: `from toolbrain import ToolBrain, ModelFactory, RewardSystem

# Test core components
brain = ToolBrain()
factory = ModelFactory()
rewards = RewardSystem()

print("✅ All core components loaded successfully!")`
    },
    {
      title: 'GPU Compatibility Test',
      code: `from toolbrain import ToolBrain
import torch

brain = ToolBrain()
device = brain.get_device()
print(f"Default device: {device}")

if torch.cuda.is_available():
    print(f"✅ GPU support enabled - {torch.cuda.get_device_name()}")
else:
    print("⚠️  CPU mode - Consider installing CUDA for better performance")`
    }
  ];

  const troubleshooting = [
    {
      issue: "ImportError: No module named 'toolbrain'",
      solutions: [
        "Ensure virtual environment is activated",
        "Reinstall with: pip install --force-reinstall toolbrain",
        "Check Python path: python -c \"import sys; print(sys.path)\""
      ]
    },
    {
      issue: "CUDA out of memory",
      solutions: [
        "Reduce batch size in your configuration",
        "Use gradient checkpointing: brain.config.use_checkpointing = True",
        "Monitor GPU usage: nvidia-smi"
      ]
    },
    {
      issue: "Permission denied during installation",
      solutions: [
        "Use --user flag: pip install --user toolbrain",
        "Run as administrator (Windows) or use sudo (Linux/Mac)",
        "Check directory permissions"
      ]
    }
  ];

  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl font-bold text-[#E6EDF3] mb-4">Installation Guide</h1>
          <p className="text-xl text-gray-400 leading-relaxed">
            Get ToolBrain up and running on your system in minutes. This guide covers 
            installation, setup, and verification steps for all supported platforms.
          </p>
        </div>

        {/* System Requirements */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">System Requirements</h2>
          
          <div className="bg-[#161B22] border border-[#30363D] rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-[#21262D]">
                  <tr>
                    <th className="text-left text-[#E6EDF3] font-semibold p-4">Component</th>
                    <th className="text-left text-[#E6EDF3] font-semibold p-4">Version</th>
                    <th className="text-left text-[#E6EDF3] font-semibold p-4">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {systemRequirements.map((req, index) => (
                    <tr key={index} className="border-t border-[#30363D]">
                      <td className="p-4 text-[#E6EDF3] font-medium">{req.name}</td>
                      <td className="p-4 text-gray-400 font-mono">{req.version}</td>
                      <td className="p-4">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          req.status === 'required' 
                            ? 'bg-[#F85149] text-white' 
                            : req.status === 'recommended'
                            ? 'bg-[#FB8500] text-white'
                            : 'bg-[#3FB950] text-white'
                        }`}>
                          {req.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Installation Steps */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Installation Steps</h2>
          
          <div className="space-y-8">
            {installationSteps.map((step, index) => (
              <div key={index} className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <div className="flex items-start mb-4">
                  <div className="flex items-center justify-center w-8 h-8 bg-[#58A6FF] text-white rounded-full font-bold text-sm mr-4 mt-1">
                    {index + 1}
                  </div>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold text-[#E6EDF3] mb-2">{step.title}</h3>
                    <p className="text-gray-400 mb-4">{step.description}</p>
                  </div>
                </div>
                
                <CodeBlock language="bash">
                  {step.code}
                </CodeBlock>
              </div>
            ))}
          </div>
        </div>

        {/* Verification */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Verify Installation</h2>
          <p className="text-gray-400 mb-6">
            Run these tests to ensure ToolBrain is installed correctly and all components are working.
          </p>
          
          <div className="space-y-6">
            {verificationTests.map((test, index) => (
              <div key={index} className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">{test.title}</h3>
                <CodeBlock language="python">
                  {test.code}
                </CodeBlock>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Test */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Quick Functionality Test</h2>
          <p className="text-gray-400 mb-6">
            Run this comprehensive test to verify all major components are working:
          </p>
          
          <CodeBlock language="python">
{`# Complete installation test
from toolbrain import ToolBrain
import sys

def test_installation():
    print("🧠 Testing ToolBrain Installation...")
    print("=" * 50)
    
    try:
        # Test 1: Basic import
        print("✅ Import successful")
        
        # Test 2: Core components
        brain = ToolBrain()
        print("✅ ToolBrain core initialized")
        
        # Test 3: Device detection
        device = brain.get_device()
        print(f"✅ Device detection: {device}")
        
        # Test 4: Simple operation
        config = brain.get_default_config()
        print("✅ Configuration system working")
        
        print("=" * 50)
        print("🎉 All tests passed! ToolBrain is ready to use.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Please check the troubleshooting section below.")

if __name__ == "__main__":
    test_installation()`}
          </CodeBlock>
        </div>

        {/* Troubleshooting */}
        <div className="mb-12">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-6">Troubleshooting</h2>
          
          <div className="space-y-6">
            {troubleshooting.map((item, index) => (
              <div key={index} className="bg-[#161B22] border border-[#30363D] rounded-lg p-6">
                <h3 className="text-lg font-semibold text-[#F85149] mb-4">❌ {item.issue}</h3>
                <div className="space-y-2">
                  <p className="text-[#E6EDF3] font-medium mb-2">Solutions:</p>
                  {item.solutions.map((solution, sIndex) => (
                    <div key={sIndex} className="flex items-start">
                      <span className="text-[#3FB950] mr-2">•</span>
                      <span className="text-gray-400">{solution}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Next Steps */}
        <div className="bg-gradient-to-r from-[#58A6FF]/10 to-[#7C3AED]/10 border border-[#58A6FF]/20 rounded-lg p-8 text-center">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-4">🚀 Ready for the Next Step?</h2>
          <p className="text-gray-300 mb-6">
            Great! ToolBrain is now installed and ready. Let's create your first intelligent agent.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button className="bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-6 py-3 rounded-lg font-medium transition-colors duration-200">
              → Quick Start Tutorial
            </button>
            <button className="border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] px-6 py-3 rounded-lg font-medium transition-colors duration-200">
              Browse Examples
            </button>
          </div>
        </div>
      </div>
    </Layout>
  );
}