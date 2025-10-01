'use client';

import Layout from '@/components/Layout';
import { useState, useEffect, useRef } from 'react';
import Image from "next/image";

export default function Introduction() {
  const [logoHovered, setLogoHovered] = useState(false);
  const [typedText, setTypedText] = useState('');
  const [showCursor, setShowCursor] = useState(true);
  const [visibleCards, setVisibleCards] = useState(new Set());
  
  const sectionRef = useRef<HTMLDivElement>(null);
  const cardRefs = useRef<(HTMLDivElement | null)[]>([]);
  
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

  // Intersection Observer for scroll animations
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const cardIndex = parseInt((entry.target as HTMLElement).dataset.cardIndex || '0');
            setVisibleCards(prev => new Set([...prev, cardIndex]));
          }
        });
      },
      {
        threshold: 0.3,
        rootMargin: '0px 0px -100px 0px'
      }
    );

    cardRefs.current.forEach((ref) => {
      if (ref) observer.observe(ref);
    });

    return () => observer.disconnect();
  }, []);
  
  const challenges = [
    {
      title: 'High Barrier to Entry',
      description: 'Complex frameworks & steep learning curves'
    },
    {
      title: 'Difficult Reward Design',
      description: 'Manual effort is slow and automated judges are inflexible'
    },
    {
      title: 'Overwhelming Tool Choice',
      description: 'Agents get lost when provided with too many tools'
    },
    {
      title: 'High Computational Cost',
      description: 'Training large, capable models is too expensive'
    }
  ];

  const solutions = [
    {
      title: 'Unified API & Architecture',
      description: 'A minimalist Brain API that abstracts away all RL complexity'
    },
    {
      title: 'Flexible, Hybrid Rewards',
      description: 'Seamlessly combine custom code with our powerful LLM-as-a-Judge'
    },
    {
      title: 'Intelligent Tool Retrieval',
      description: 'Automatically selects only the most relevant tools for the agent'
    },
    {
      title: 'Efficient & Accessible Training',
      description: 'Integrated Unsloth & QLoRA make training large models practical'
    }
  ];

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        {/* SECTION 1: HERO SECTION */}
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
          
          <p className="text-xl text-gray-400 mb-10 max-w-3xl mx-auto leading-relaxed">
            ToolBrain is a lightweight framework that lets you train your existing tool-using agents with RL, 
            using a powerful and intuitive API.
          </p>

          {/* Primary CTA */}
          <div className="mb-8">
            <a 
              href="/get-started/quickstart"
              className="inline-block bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-10 py-4 rounded-lg text-lg font-semibold transition-colors duration-200 shadow-lg hover:shadow-xl"
            >
              Get Started
            </a>
          </div>

          {/* Secondary CTAs */}
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

        {/* SECTION 2: PROBLEM-SOLUTION SECTION */}
        <div className="mb-20" ref={sectionRef}>
          <h2 className="text-4xl font-bold text-[#E6EDF3] text-center mb-16">
            Common Challenges & Our Solutions
          </h2>
          
          {/* Individual Problem-Solution Pairs */}
          <div className="space-y-12">
            {challenges.map((challenge, index) => (
              <div 
                key={index}
                className="relative grid lg:grid-cols-2 gap-8 lg:gap-16 items-center"
              >
                {/* Problem Card */}
                <div 
                  ref={el => { cardRefs.current[index * 2] = el; }}
                  data-card-index={index * 2}
                  className={`transform transition-all duration-700 ease-out ${
                    visibleCards.has(index * 2) 
                      ? 'translate-x-0 opacity-100' 
                      : '-translate-x-12 opacity-0'
                  }`}
                  style={{ transitionDelay: `${index * 200}ms` }}
                >
                  <div className="relative">
                    <div className="absolute -top-4 -left-4 w-8 h-8 bg-[#F85149] rounded-full flex items-center justify-center text-white font-bold text-sm">
                      {index + 1}
                    </div>
                    <div className="bg-[#2D1B1B] border border-[#F85149]/30 rounded-lg p-6 hover:border-[#F85149]/50 transition-all duration-300 hover:transform hover:scale-105">
                      <div className="flex items-start gap-3 mb-3">
                        <div className="text-2xl">❌</div>
                        <h4 className="text-lg font-semibold text-[#F85149]">
                          {challenge.title}
                        </h4>
                      </div>
                      <p className="text-gray-300 leading-relaxed pl-11">
                        {challenge.description}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Connecting Arrow */}
                <div className="hidden lg:flex absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 z-10">
                  <div className="flex items-center">
                    <div className="w-8 h-0.5 bg-gradient-to-r from-[#F85149] to-[#3FB950]"></div>
                    <div className="w-0 h-0 border-l-[8px] border-l-[#3FB950] border-t-[6px] border-t-transparent border-b-[6px] border-b-transparent ml-1"></div>
                  </div>
                </div>

                {/* Solution Card */}
                <div 
                  ref={el => { cardRefs.current[index * 2 + 1] = el; }}
                  data-card-index={index * 2 + 1}
                  className={`transform transition-all duration-700 ease-out ${
                    visibleCards.has(index * 2 + 1) 
                      ? 'translate-x-0 opacity-100' 
                      : 'translate-x-12 opacity-0'
                  }`}
                  style={{ transitionDelay: `${index * 200 + 400}ms` }}
                >
                  <div className="relative">
                    <div className="absolute -top-4 -right-4 w-8 h-8 bg-[#3FB950] rounded-full flex items-center justify-center text-white font-bold text-sm">
                      {index + 1}
                    </div>
                    <div className="bg-[#1B2D1B] border border-[#3FB950]/30 rounded-lg p-6 hover:border-[#3FB950]/50 transition-all duration-300 hover:transform hover:scale-105">
                      <div className="flex items-start gap-3 mb-3">
                        <div className="text-2xl">✅</div>
                        <h4 className="text-lg font-semibold text-[#3FB950]">
                          {solutions[index].title}
                        </h4>
                      </div>
                      <p className="text-gray-300 leading-relaxed pl-11">
                        {solutions[index].description}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* SECTION 3: FINAL CALL TO ACTION */}
        <div className="bg-gradient-to-r from-[#161B22] to-[#21262D] border border-[#30363D] rounded-xl p-12 text-center">
          <h2 className="text-3xl font-bold text-[#E6EDF3] mb-6">
            Ready to Build Smarter Agents?
          </h2>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Follow our Quickstart guide and train your first agent in less than 5 minutes.
          </p>
          
          <a 
            href="/get-started/quickstart"
            className="inline-block bg-[#58A6FF] hover:bg-[#4A90E2] text-white px-10 py-4 rounded-lg text-lg font-semibold transition-colors duration-200 shadow-lg hover:shadow-xl"
          >
            Start the Quickstart Guide
          </a>
        </div>
      </div>
    </Layout>
  );
}
