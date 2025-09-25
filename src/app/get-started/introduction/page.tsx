'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';
import { useState, useEffect } from 'react';
import Image from "next/image";

export default function Introduction() {
  const [logoHovered, setLogoHovered] = useState(false);
  const [typedText, setTypedText] = useState('');
  const fullText = 'AI-Powered Agent Training Framework';

  useEffect(() => {
    let i = 0;
    const typing = setInterval(() => {
      if (i < fullText.length) {
        setTypedText(fullText.slice(0, i + 1));
        i++;
      } else {
        clearInterval(typing);
      }
    }, 100);

    return () => clearInterval(typing);
  }, []);

  const challenges = [
    {
      icon: '🧠',
      title: 'Complex Agent Training',
      problem: 'Manual hyperparameter tuning is time-consuming and error-prone',
      solution: 'Automated optimization with intelligent search algorithms'
    },
    {
      icon: '⚡',
      title: 'Performance Bottlenecks',
      problem: 'Slow training cycles and inefficient resource utilization',
      solution: 'Optimized training loops with parallel processing capabilities'
    },
    {
      icon: '📊',
      title: 'Reward Engineering',
      problem: 'Designing effective reward functions requires deep expertise',
      solution: 'Built-in reward templates and automated reward shaping'
    },
    {
      icon: '🔧',
      title: 'Model Selection',
      problem: 'Choosing the right model architecture is challenging',
      solution: 'Smart model recommendations based on task requirements'
    }
  ];

  const features = [
    {
      title: 'Intelligent Hyperparameter Optimization',
      description: 'Advanced algorithms automatically find optimal parameters',
      code: `brain = ToolBrain()
brain.optimize_hyperparameters(
    learning_rate_range=(0.001, 0.1),
    batch_size_options=[16, 32, 64, 128],
    optimizer_types=['adam', 'sgd', 'rmsprop']
)`
    },
    {
      title: 'Flexible Reward Systems',
      description: 'Design custom reward functions or use pre-built templates',
      code: `reward_system = RewardSystem()
reward_system.add_component(
    'task_completion', weight=0.7
)
reward_system.add_component(
    'efficiency_bonus', weight=0.3
)`
    },
    {
      title: 'Model Factory Pattern',
      description: 'Easy model creation and management with factory pattern',
      code: `factory = ModelFactory()
model = factory.create_model(
    model_type='transformer',
    config={
        'hidden_size': 768,
        'num_layers': 12,
        'attention_heads': 12
    }
)`
    }
  ];

  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <div 
            className={`inline-block mb-6 transition-transform duration-300 ${logoHovered ? 'scale-110' : ''}`}
            onMouseEnter={() => setLogoHovered(true)}
            onMouseLeave={() => setLogoHovered(false)}
          >
            <div className="text-6xl font-bold bg-gradient-to-r from-[#58A6FF] to-[#7C3AED] bg-clip-text text-transparent">
              ToolBrain
            </div>
          </div>
          
          <h1 className="text-2xl text-[#E6EDF3] mb-4 h-8">
            {typedText}
            <span className="animate-pulse">|</span>
          </h1>
          
          <p className="text-lg text-gray-400 mb-8 max-w-2xl mx-auto leading-relaxed">
            ToolBrain revolutionizes AI agent development with automated hyperparameter optimization, 
            intelligent reward systems, and streamlined model training workflows. Build smarter agents faster.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-6">
            <button className="bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-8 py-3 rounded-lg font-medium transition-colors duration-200">
              Get Started
            </button>
            <button className="border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] px-8 py-3 rounded-lg font-medium transition-colors duration-200">
              View Examples
            </button>
          </div>

          {/* Additional Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <a 
              href="https://github.com/toolbrain/toolbrain" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] hover:text-[#58A6FF] px-6 py-2.5 rounded-lg font-medium transition-colors duration-200"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              View on GitHub
            </a>
            
            {/* arXiv */}
            <a 
                href="https://arxiv.org/abs/2024.toolbrain" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] hover:text-[#58A6FF] px-6 py-2.5 rounded-lg font-medium transition-colors duration-200"
            >
                <Image 
                src="/arxiv.jpg" 
                alt="arXiv Logo" 
                width={20} 
                height={20} 
                />
                Read the Paper
            </a>

            {/* YouTube */}
            <a 
                href="https://www.youtube.com/watch?v=8kEEV-vYjF8" 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] hover:text-[#58A6FF] px-6 py-2.5 rounded-lg font-medium transition-colors duration-200"
            >
                <Image 
                src="/youtube.png" 
                alt="YouTube Logo" 
                width={20} 
                height={20} 
                />
                Video Demo
            </a>
          </div>
        </div>

        {/* Challenges & Solutions Grid */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-[#E6EDF3] text-center mb-12">
            Common Challenges & Our Solutions
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            {challenges.map((item, index) => (
              <div key={index} className="bg-[#161B22] border border-[#30363D] rounded-lg p-6 hover:border-[#58A6FF] transition-colors duration-300">
                <div className="text-3xl mb-4">{item.icon}</div>
                <h3 className="text-xl font-semibold text-[#E6EDF3] mb-3">{item.title}</h3>
                
                <div className="mb-4">
                  <div className="text-sm text-[#F85149] font-medium mb-1">❌ Problem:</div>
                  <p className="text-gray-400 text-sm">{item.problem}</p>
                </div>
                
                <div>
                  <div className="text-sm text-[#3FB950] font-medium mb-1">✅ Solution:</div>
                  <p className="text-gray-300 text-sm">{item.solution}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* API Showcase */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-[#E6EDF3] text-center mb-12">
            Powerful & Intuitive API
          </h2>
          
          <div className="space-y-12">
            {features.map((feature, index) => (
              <div key={index} className="grid lg:grid-cols-2 gap-8 items-center">
                <div className={index % 2 === 1 ? 'lg:order-2' : ''}>
                  <h3 className="text-2xl font-semibold text-[#E6EDF3] mb-4">{feature.title}</h3>
                  <p className="text-gray-400 text-lg leading-relaxed">{feature.description}</p>
                </div>
                
                <div className={index % 2 === 1 ? 'lg:order-1' : ''}>
                  <CodeBlock language="python">
                    {feature.code}
                  </CodeBlock>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Start Preview */}
        <div className="bg-[#161B22] border border-[#30363D] rounded-lg p-8 text-center">
          <h2 className="text-2xl font-bold text-[#E6EDF3] mb-4">
            Ready to Build Intelligent Agents?
          </h2>
          <p className="text-gray-400 mb-6">
            Get started with ToolBrain in just a few minutes
          </p>
          
          <CodeBlock language="bash">
{`# Install ToolBrain
pip install toolbrain

# Create your first intelligent agent
from toolbrain import ToolBrain

brain = ToolBrain()
agent = brain.create_agent(
    task_type='reinforcement_learning',
    environment='custom_env'
)

# Start training with automatic optimization
brain.train(agent, auto_optimize=True)`}
          </CodeBlock>
          
          <div className="mt-6">
            <button className="bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-6 py-2 rounded-lg font-medium transition-colors duration-200 mr-4">
              Installation Guide
            </button>
            <button className="border border-[#30363D] hover:border-[#58A6FF] text-[#E6EDF3] px-6 py-2 rounded-lg font-medium transition-colors duration-200">
              Full Tutorial
            </button>
          </div>
        </div>
      </div>
    </Layout>
  );
}