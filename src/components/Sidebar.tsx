'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import Image from "next/image";

interface NavItem {
  name: string;
  href?: string;
  children?: NavItem[];
}

const navigation: NavItem[] = [
  {
    name: 'GET STARTED',
    children: [
      { name: 'Introduction', href: '/get-started/introduction' },
      { name: 'Installation', href: '/get-started/installation' },
    ]
  },
  {
    name: 'TUTORIALS',
    children: [
      { name: 'Solving HPO with RL', href: '/tutorials/hpo-with-rl' },
      { name: 'Training an Email Search Agent', href: '/tutorials/email-search-agent' },
    ]
  },
  {
    name: 'KEY FEATURES',
    children: [
      { name: 'Unified API & Architecture', href: '/key-features/unified-api' },
      { name: 'Flexible Rewards', href: '/key-features/flexible-rewards' },
      { name: 'Multiple Learning Algorithms', href: '/key-features/multiple-algorithms' },
      { name: 'Knowledge Distillation', href: '/key-features/knowledge-distillation' },
      { name: 'Intelligent Tool Retrieval', href: '/key-features/intelligent-tool-retrieval' },
      { name: 'Zero-Learn Task Generation', href: '/key-features/zero-learn' },
      { name: 'Efficient Training', href: '/key-features/efficient-training' },
    ]
  },
  {
    name: 'CONCEPTUAL GUIDES',
    children: [
      { name: 'The "Coach-Athlete" Paradigm', href: '/conceptual-guides/coach-athlete-paradigm' },
      { name: 'The Execution Trace', href: '/conceptual-guides/execution-trace' },
    ]
  },
  {
    name: 'RESOURCES',
    children: [
      { name: 'API Reference', href: '/resources/api-reference' },
    ]
  }
];

interface SidebarProps {
  className?: string;
}

export default function Sidebar({ className = '' }: SidebarProps) {
  const pathname = usePathname();
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['GET STARTED', 'TUTORIALS', 'KEY FEATURES', 'CONCEPTUAL GUIDES', 'RESOURCES'])
  );

  const toggleSection = (sectionName: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(sectionName)) {
      newExpanded.delete(sectionName);
    } else {
      newExpanded.add(sectionName);
    }
    setExpandedSections(newExpanded);
  };

  return (
    <nav className={`fixed top-0 left-0 z-40 w-80 h-screen bg-[#0D1117] border-r border-gray-700 overflow-y-auto ${className}`}>
      <div className="p-6">
        {/* Logo and Title */}
        <div className="mb-8 flex items-center gap-3">
          <Image
                src="/toolbrain.png"
                alt="ToolBrain Logo"
                width={96}  
                height={96}  
                className="rounded-full"
            />
          <div>
            <h1 className="text-xl font-bold text-[#E6EDF3]">ToolBrain</h1>
            <p className="text-sm text-gray-400">RL for Agentic Systems</p>
          </div>
        </div>

        {/* Navigation */}
        <div className="space-y-1">
          {navigation.map((section) => (
            <div key={section.name}>
              {section.children ? (
                <>
                  <button
                    onClick={() => toggleSection(section.name)}
                    className="w-full flex items-center justify-between p-2 text-left text-sm font-semibold text-gray-300 hover:text-[#58A6FF] transition-colors"
                  >
                    <span>{section.name}</span>
                    <svg
                      className={`w-4 h-4 transition-transform ${
                        expandedSections.has(section.name) ? 'rotate-90' : ''
                      }`}
                      fill="currentColor"
                      viewBox="0 0 20 20"
                    >
                      <path
                        fillRule="evenodd"
                        d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </button>
                  {expandedSections.has(section.name) && (
                    <div className="ml-2 space-y-1">
                      {section.children.map((item) => (
                        <Link
                          key={item.name}
                          href={item.href!}
                          className={`block p-2 pl-4 text-sm rounded-md transition-colors ${
                            pathname === item.href
                              ? 'bg-[#58A6FF]/10 text-[#58A6FF] border-l-2 border-[#58A6FF]'
                              : 'text-gray-400 hover:text-[#E6EDF3] hover:bg-[#161B22]'
                          }`}
                        >
                          {item.name}
                        </Link>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                <Link
                  href={section.href!}
                  className={`block p-2 text-sm rounded-md transition-colors ${
                    pathname === section.href
                      ? 'bg-[#58A6FF]/10 text-[#58A6FF]'
                      : 'text-gray-400 hover:text-[#E6EDF3] hover:bg-[#161B22]'
                  }`}
                >
                  {section.name}
                </Link>
              )}
            </div>
          ))}
        </div>
      </div>
    </nav>
  );
}