'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import { useState, useEffect } from 'react';
import Image from "next/image";

export default function Introduction() {
  const [logoHovered, setLogoHovered] = useState(false);
  const [typedText, setTypedText] = useState('');
  const [showCursor, setShowCursor] = useState(true);
  
  const fullText = 'Reinforcement Learning for Agents.\nMade Simple.';

  useEffect(() => {
    let i = 0;
    const typing = setInterval(() => {
      if (i < fullText.length) {
        setTypedText(fullText.slice(0, i + 1));
        i++;
      } else {
        clearInterval(typing);
        // Hide cursor after typing is complete
        setTimeout(() => setShowCursor(false), 1000);
      }
    }, 80);

    return () => clearInterval(typing);
  }, []);
  
  const scrollToQuickstart = () => {
    const quickstartSection = document.getElementById('quickstart-section');
    if (quickstartSection) {
      quickstartSection.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const quickstartCode = `# 1. Imports and Component Definition
from toolbrain import Brain
from toolbrain.rewards import reward_exact_match
from smolagents import CodeAgent, TransformersModel, tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

# Define a standard agent (CPU-compatible)
model = TransformersModel("Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens=128)
agent = CodeAgent(model=model, tools=[multiply], max_steps=1)

# 2. Initialize the Brain with your agent and a built-in reward
brain = Brain(
    agent=agent,
    reward_func=reward_exact_match,
    learning_algorithm="GRPO"
)

# 3. Define a task and start training!
dataset = [{"query": "What is 8 multiplied by 7?", "gold_answer": "56"}]
brain.train(dataset, num_iterations=10)

# 4. Save your trained agent
brain.save("./my_first_trained_agent")`;

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        {/* PHẦN 1: HERO SECTION */}
        <div className="text-center mb-20">
          <div 
            className={`inline-block mb-8 transition-transform duration-300 ${logoHovered ? 'scale-110' : ''}`}
            onMouseEnter={() => setLogoHovered(true)}
            onMouseLeave={() => setLogoHovered(false)}
          >
            <div className="text-7xl font-bold bg-gradient-to-r from-[#58A6FF] to-[#7C3AED] bg-clip-text text-transparent">
              ToolBrain
            </div>
          </div>
          
          {/* Tiêu đề Lớn */}
          <h1 className="text-4xl md:text-5xl font-bold text-[#E6EDF3] mb-6 leading-tight min-h-[140px] flex items-center justify-center">
            <span>
              {typedText.split('\n').map((line, index) => (
                <span key={index}>
                  {index === 0 && line}
                  {index === 1 && (
                    <span className="text-[#58A6FF]">{line}</span>
                  )}
                  {index === 0 && typedText.includes('\n') && <br />}
                </span>
              ))}
              {showCursor && <span className="animate-pulse text-[#58A6FF]">|</span>}
            </span>
          </h1>
          
          {/* Dòng phụ */}
          <p className="text-xl text-gray-400 mb-10 max-w-3xl mx-auto leading-relaxed">
            ToolBrain is a lightweight framework that lets you train your existing tool-using agents with RL, 
            using a powerful and intuitive API.
          </p>

          {/* Nút chính (Primary Button) */}
          <div className="mb-8">
            <button 
              onClick={scrollToQuickstart}
              className="inline-block bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-10 py-4 rounded-lg text-lg font-semibold transition-colors duration-200 shadow-lg hover:shadow-xl"
            >
              Get Started Below
            </button>
          </div>

          {/* Các nút phụ (Secondary Buttons) */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a 
              href="https://github.com/toolbrain/toolbrain" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-3 border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] hover:text-[#58A6FF] px-6 py-3 rounded-lg font-medium transition-colors duration-200"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              View on GitHub
            </a>
            
            <a 
              href="https://arxiv.org/abs/2024.toolbrain" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-3 border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] hover:text-[#58A6FF] px-6 py-3 rounded-lg font-medium transition-colors duration-200"
            >
              <Image 
                src="/arxiv.jpg" 
                alt="arXiv Logo" 
                width={20} 
                height={20} 
              />
              Read the Paper
            </a>

            <a 
              href="https://www.youtube.com/watch?v=8kEEV-vYjF8" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-3 border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] hover:text-[#58A6FF] px-6 py-3 rounded-lg font-medium transition-colors duration-200"
            >
              <Image 
                src="/youtube.png" 
                alt="YouTube Logo" 
                width={20} 
                height={20} 
              />
              Watch the Demo
            </a>
          </div>
        </div>

        {/* PHẦN 2: QUICKSTART - "NGÔI SAO" CỦA TRANG */}
        <div id="quickstart-section" className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] text-center mb-6">
            Get Started in Minutes
          </h2>
          
          <p className="text-xl text-gray-400 mb-10 max-w-4xl mx-auto text-center leading-relaxed">
            We believe in "show, don't tell". The code below is a complete, runnable example. 
            It trains a simple agent to use a multiply tool. Copy, paste, and run it to see ToolBrain in action.
          </p>

          <div className="max-w-4xl mx-auto">
            <CodeBlock language="python" filename="quickstart_example.py">
              {quickstartCode}
            </CodeBlock>
          </div>
        </div>

        {/* PHẦN 3: "WHAT'S UNDER THE HOOD?" */}
        <div className="mb-20">
          <h2 className="text-4xl font-bold text-[#E6EDF3] text-center mb-6">
            The Power Behind the Simplicity
          </h2>
          
          {/* Grid 2x2 */}
          <div className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto">
            {/* Ô 1: Unified API & Architecture */}
            <div className="bg-gradient-to-br from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8 hover:border-[#58A6FF]/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="text-3xl mb-4">🧠</div>
              <h3 className="text-xl font-bold text-[#58A6FF] mb-4">
                Unified API & Architecture
              </h3>
              <p className="text-gray-400 leading-relaxed">
                The Brain class abstracts away the entire RL loop. You focus on your agent, we handle the training.
              </p>
            </div>

            {/* Ô 2: Flexible, Hybrid Rewards */}
            <div className="bg-gradient-to-br from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8 hover:border-[#58A6FF]/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="text-3xl mb-4">🎯</div>
              <h3 className="text-xl font-bold text-[#58A6FF] mb-4">
                Flexible, Hybrid Rewards
              </h3>
              <p className="text-gray-400 leading-relaxed">
                Use our built-in rewards like reward_exact_match, write your own Python functions, or leverage our powerful LLM-as-a-Judge.
              </p>
            </div>

            {/* Ô 3: Intelligent Tool & Data Management */}
            <div className="bg-gradient-to-br from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8 hover:border-[#58A6FF]/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="text-3xl mb-4">🔧</div>
              <h3 className="text-xl font-bold text-[#58A6FF] mb-4">
                Intelligent Tool & Data Management
              </h3>
              <p className="text-gray-400 leading-relaxed">
                Automatically select relevant tools with Tool Retrieval and generate new training tasks with Zero-Learn.
              </p>
            </div>

            {/* Ô 4: Efficient & Advanced Training */}
            <div className="bg-gradient-to-br from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-8 hover:border-[#58A6FF]/50 transition-all duration-300 hover:transform hover:scale-105">
              <div className="text-3xl mb-4">⚡</div>
              <h3 className="text-xl font-bold text-[#58A6FF] mb-4">
                Efficient & Advanced Training
              </h3>
              <p className="text-gray-400 leading-relaxed">
                Out-of-the-box support for GRPO, DPO, and Knowledge Distillation, all accelerated by Unsloth and QLoRA.
              </p>
            </div>
          </div>
        </div>

        {/* Call to Action cuối cùng */}
        <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12 text-center">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            Ready to dive deeper?
          </h2>
          
          <a 
            href="/key-features/unified-api"
            className="inline-block bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-10 py-4 rounded-lg text-lg font-semibold transition-colors duration-200 shadow-lg hover:shadow-xl"
          >
            Explore Full Documentation
          </a>
        </div>
      </div>
    </Layout>
  );
}
