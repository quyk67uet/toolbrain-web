'use client';

import Image from "next/image";
import { useEffect, useRef } from "react";

// Icon Components
const GitHubIcon = () => (
  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
    <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
  </svg>
);

const PaperIcon = () => (
  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
    <path d="M9 2a1 1 0 000 2h6a1 1 0 100-2H9z" />
    <path fillRule="evenodd" d="M4 5a2 2 0 012-2h1a1 1 0 000 2H6v6.586l.293-.293a1 1 0 011.414 1.414l-2 2a1 1 0 01-1.414 0l-2-2a1 1 0 111.414-1.414L4 11.586V5zm9 0a2 2 0 012-2h1a1 1 0 100 2h-1v6.586l.293-.293a1 1 0 011.414 1.414l-2 2a1 1 0 01-1.414 0l-2-2a1 1 0 111.414-1.414L13 11.586V5z" clipRule="evenodd" />
  </svg>
);

// Animated Background Component
const AnimatedBackground = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const shapes: Array<{
      x: number;
      y: number;
      size: number;
      speed: number;
      opacity: number;
      angle: number;
    }> = [];

    // Create animated shapes
    for (let i = 0; i < 15; i++) {
      shapes.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 3 + 1,
        speed: Math.random() * 0.5 + 0.1,
        opacity: Math.random() * 0.5 + 0.1,
        angle: Math.random() * Math.PI * 2,
      });
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      shapes.forEach((shape) => {
        shape.x += Math.cos(shape.angle) * shape.speed;
        shape.y += Math.sin(shape.angle) * shape.speed;
        
        // Wrap around edges
        if (shape.x > canvas.width) shape.x = 0;
        if (shape.x < 0) shape.x = canvas.width;
        if (shape.y > canvas.height) shape.y = 0;
        if (shape.y < 0) shape.y = canvas.height;
        
        ctx.beginPath();
        ctx.arc(shape.x, shape.y, shape.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(88, 166, 255, ${shape.opacity})`;
        ctx.fill();
      });
      
      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 z-0"
      style={{ background: 'linear-gradient(135deg, #0D1117 0%, #161B22 100%)' }}
    />
  );
};

// Star Animation Component for Logo Background
const StarBackground = () => {
  const starCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = starCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = 300;
    canvas.height = 300;

    const stars: Array<{
      x: number;
      y: number;
      size: number;
      twinkleSpeed: number;
      brightness: number;
      phase: number;
    }> = [];

    // Create twinkling stars
    for (let i = 0; i < 30; i++) {
      stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: Math.random() * 2 + 0.5,
        twinkleSpeed: Math.random() * 0.02 + 0.01,
        brightness: Math.random(),
        phase: Math.random() * Math.PI * 2,
      });
    }

    const animateStars = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      stars.forEach((star) => {
        star.phase += star.twinkleSpeed;
        star.brightness = (Math.sin(star.phase) + 1) / 2;
        
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${star.brightness * 0.8})`;
        ctx.fill();
        
        // Add cross sparkle effect
        if (star.brightness > 0.7) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(255, 255, 255, ${star.brightness * 0.6})`;
          ctx.lineWidth = 1;
          ctx.moveTo(star.x - star.size * 3, star.y);
          ctx.lineTo(star.x + star.size * 3, star.y);
          ctx.moveTo(star.x, star.y - star.size * 3);
          ctx.lineTo(star.x, star.y + star.size * 3);
          ctx.stroke();
        }
      });
      
      requestAnimationFrame(animateStars);
    };

    animateStars();
  }, []);

  return (
    <canvas
      ref={starCanvasRef}
      className="absolute inset-0 z-0"
      style={{ width: '100%', height: '100%' }}
    />
  );
};

// Enhanced Image Frame Component
const ImageFrame = ({ 
  src, 
  alt, 
  width, 
  height, 
  className = "" 
}: { 
  src: string; 
  alt: string; 
  width: number; 
  height: number; 
  className?: string; 
}) => (
  <div className="relative group">
    <div className="absolute inset-0 bg-gradient-to-r from-[#58A6FF] to-[#79C0FF] rounded-lg blur-sm opacity-75 group-hover:opacity-100 transition-opacity duration-300"></div>
    <div className="relative bg-[#0D1117] p-4 rounded-lg border border-[#58A6FF]/50 group-hover:border-[#58A6FF] transition-colors duration-300">
      <Image
        src={src}
        alt={alt}
        width={width}
        height={height}
        className={`rounded-lg shadow-2xl ${className}`}
      />
    </div>
  </div>
);

// Problem Card Component
const ProblemCard = ({ title, text, icon }: { title: string; text: string; icon: string }) => (
  <div className="bg-[#161B22] border border-gray-700 rounded-lg p-6 hover:border-[#58A6FF] transition-colors duration-300">
    <div className="text-4xl mb-4">{icon}</div>
    <h3 className="text-lg font-semibold text-[#E6EDF3] mb-3">{title}</h3>
    <p className="text-gray-400 text-sm">{text}</p>
  </div>
);

export default function Home() {
  return (
    <div className="min-h-screen bg-[#0D1117] text-[#E6EDF3] font-sans overflow-x-hidden">
      {/* Hero Section */}
      <section id="hero" className="relative min-h-screen flex items-center justify-center px-4">
        <AnimatedBackground />
        <div className="relative z-10 text-center max-w-4xl mx-auto">
          <div className="mb-8 flex justify-center">
            {/* Enhanced Logo Container with Star Background */}
            <div className="relative w-64 h-64 rounded-full bg-gradient-to-br from-[#161B22] via-[#0D1117] to-[#161B22] border-2 border-[#58A6FF]/30 shadow-2xl overflow-hidden group hover:border-[#58A6FF] transition-colors duration-500">
              <StarBackground />
              <div className="relative z-10 flex items-center justify-center w-full h-full">
                <Image
                  src="/toolbrain.png"
                  alt="ToolBrain Logo"
                  width={160}
                  height={160}
                  className="drop-shadow-2xl group-hover:scale-110 transition-transform duration-500"
                  priority
                />
              </div>
              {/* Glow effect */}
              <div className="absolute inset-0 rounded-full bg-gradient-to-r from-[#58A6FF]/20 to-[#79C0FF]/20 blur-xl opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            </div>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-[#58A6FF] to-[#79C0FF] bg-clip-text text-transparent">
            ToolBrain: A Flexible Reinforcement Learning Framework for Agentic Systems
          </h1>
          <h2 className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Go from a failing, unreliable agent to a proficient, tool-using expert. ToolBrain makes the power of RL for agents Easy-to-Use, Flexible, and Efficient.
          </h2>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="#"
              className="inline-flex items-center px-6 py-3 bg-[#58A6FF] text-white font-semibold rounded-lg hover:bg-[#4A9AFF] transition-colors duration-300 gap-2"
            >
              <GitHubIcon />
              View on GitHub
            </a>
            <a
              href="#"
              className="inline-flex items-center px-6 py-3 border border-[#58A6FF] text-[#58A6FF] font-semibold rounded-lg hover:bg-[#58A6FF] hover:text-white transition-colors duration-300 gap-2"
            >
              <PaperIcon />
              Read the Paper
            </a>
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section id="problem" className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-12">
            Training Tool-Using Agents is Broken
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ProblemCard
              title="High Barrier to Entry"
              text="Existing RL frameworks are complex and require deep expertise."
              icon="🚧"
            />
            <ProblemCard
              title="Difficult Reward Design"
              text="Manual reward engineering is slow, while automated judges are often inflexible."
              icon="⚖️"
            />
            <ProblemCard
              title="Overwhelming Tool Choice"
              text="Agents get lost and perform poorly when provided with too many irrelevant tools."
              icon="🔧"
            />
            <ProblemCard
              title="Prohibitive Computational Cost"
              text="Training capable, large models is too expensive and memory-intensive."
              icon="💻"
            />
          </div>
        </div>
      </section>

      {/* Solution Diagram Section */}
      <section id="solution-diagram" className="py-20 px-4 bg-[#161B22]">
        <div className="max-w-6xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            The ToolBrain Way: Our Contributions
          </h2>
          <p className="text-xl text-gray-300 mb-12 max-w-3xl mx-auto">
            Our entire, powerful workflow is expressed in a single, intuitive block of code.
          </p>
          <div className="max-w-4xl mx-auto">
            <ImageFrame
              src="/code_diagram.jpg"
              alt="ToolBrain Code Diagram"
              width={800}
              height={600}
              className="mx-auto"
            />
          </div>
        </div>
      </section>

      {/* Demonstration Section */}
      <section id="demonstration" className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-6">
              Demonstration: From Unreliable to Proficient
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              We demonstrate ToolBrain by training an agent on a challenging email search task.
            </p>
          </div>

          {/* Before Section */}
          <div className="mb-16">
            <h3 className="text-2xl font-bold mb-8 text-center">
              Baseline: An Untrained Agent is Unreliable
            </h3>
            <div className="max-w-4xl mx-auto">
              <ImageFrame
                src="/error.jpg"
                alt="Agent Error Screenshot"
                width={800}
                height={500}
                className="mx-auto mb-6"
              />
              <p className="text-gray-300 text-center">
                Without training, the agent is unreliable. It frequently fails with critical errors like the basic syntax error shown here, making it unusable for any real-world application.
              </p>
            </div>
          </div>

          {/* After Section */}
          <div>
            <h3 className="text-2xl font-bold mb-8 text-center">
              After Training with ToolBrain: A Proficient & Reliable Agent
            </h3>
            
            {/* Video Section */}
            <div className="mb-12">
              <div className="max-w-4xl mx-auto">
                <div className="relative aspect-video bg-black rounded-lg overflow-hidden group">
                  {/* Extended top overlay to hide person in video thumbnail */}
                  <div className="absolute top-0 left-0 w-full h-40 bg-gradient-to-b from-[#0D1117] to-transparent z-10"></div>
                  <video
                    controls
                    className="w-full h-full"
                    poster="/toolbrain.png"
                  >
                    <source src="/demo.mp4" type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                </div>
                <p className="text-gray-300 text-center mt-6">
                  This improvement is confirmed by our qualitative demo. The trained agent now intelligently navigates a multi-step task, even recovering from errors to find the correct answer.
                </p>
              </div>
            </div>

            {/* Learning Curve Section */}
            <div>
              <div className="max-w-4xl mx-auto">
                <ImageFrame
                  src="/learning_curve.jpg"
                  alt="Learning Curve"
                  width={800}
                  height={500}
                  className="mx-auto mb-6"
                />
                <p className="text-gray-300 text-center">
                  Our quantitative results show both 3B and 7B models learning effectively. The larger 7B model exhibits some zero-shot capability, but for both, ToolBrain significantly improves performance, reaching a peak correctness of over 43%.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section id="team" className="py-20 px-4 bg-[#161B22]">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-12">Meet the Team</h2>
          <div className="bg-[#0D1117] rounded-lg p-8 border border-gray-700">
            <div className="space-y-4 text-left">
              <p className="text-lg">
                <span className="font-semibold text-[#58A6FF]">Quy Minh Le¹</span>, 
                <span className="font-semibold text-[#58A6FF]"> Minh Sao Khue Luu¹</span>, 
                <span className="font-semibold text-[#58A6FF]"> Khanh-Tung Tran²</span>, 
                <span className="font-semibold text-[#58A6FF]"> Duc-Hai Nguyen²</span>, 
                <span className="font-semibold text-[#58A6FF]"> Hoang-Quoc-Viet Pham¹</span>, 
                <span className="font-semibold text-[#58A6FF]"> Quan Le³</span>, 
                <span className="font-semibold text-[#58A6FF]"> Hoang Thanh Lam⁴</span>, 
                <span className="font-semibold text-[#58A6FF]"> Hoang D. Nguyen²</span>
              </p>
              <div className="text-sm text-gray-400 space-y-2">
                <p>¹ ToolBrain Research</p>
                <p>² University College Cork</p>
                <p>³ CeADAR, UCD</p>
                <p>⁴ IBM Research</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Footer Section */}
      <section id="cta-footer" className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Start Building Smarter Agents Today
          </h2>
          <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Explore the source code, read our in-depth paper, and see how ToolBrain can accelerate your agent development.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a
              href="#"
              className="inline-flex items-center px-8 py-4 bg-[#58A6FF] text-white font-semibold rounded-lg hover:bg-[#4A9AFF] transition-colors duration-300 gap-3 text-lg"
            >
              <GitHubIcon />
              View on GitHub
            </a>
            <a
              href="#"
              className="inline-flex items-center px-8 py-4 border border-[#58A6FF] text-[#58A6FF] font-semibold rounded-lg hover:bg-[#58A6FF] hover:text-white transition-colors duration-300 gap-3 text-lg"
            >
              <PaperIcon />
              Read the Full Paper
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-gray-700">
        <div className="max-w-6xl mx-auto text-center text-gray-400">
          <p>&copy; 2025 ToolBrain Research. Building the future of agentic AI.</p>
        </div>
      </footer>
    </div>
  );
}
