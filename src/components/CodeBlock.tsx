'use client';

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState } from 'react';

interface CodeBlockProps {
  children: string;
  language?: string;
  filename?: string;
  className?: string;
}

export default function CodeBlock({ 
  children, 
  language = 'python', 
  filename,
  className = '' 
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(children.trim());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000); // Hide after 2 seconds
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };
  return (
    <div className={`relative rounded-lg overflow-hidden bg-[#161B22] border border-gray-700 ${className}`}>
      {filename && (
        <div className="px-4 py-2 bg-[#0D1117] border-b border-gray-700 text-sm text-gray-400">
          {filename}
        </div>
      )}
      <div className="relative">
        <SyntaxHighlighter
          language={language}
          style={oneDark}
          customStyle={{
            background: '#161B22',
            padding: '1.5rem',
            margin: 0,
            fontSize: '0.875rem',
            lineHeight: '1.5',
          }}
          showLineNumbers={false}
          wrapLines
          wrapLongLines
        >
          {children.trim()}
        </SyntaxHighlighter>
        
        {/* Copy button */}
        <button
          onClick={handleCopy}
          className={`absolute top-3 right-3 p-2 rounded-md border transition-all duration-200 ${
            copied 
              ? 'bg-[#3FB950] border-[#3FB950] text-white' 
              : 'bg-[#0D1117] hover:bg-[#21262D] border-gray-600 text-gray-400'
          }`}
          title={copied ? "Copied!" : "Copy to clipboard"}
        >
          {copied ? (
            // Checkmark icon
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path 
                fillRule="evenodd" 
                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" 
                clipRule="evenodd" 
              />
            </svg>
          ) : (
            // Copy icon
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M8 3a1 1 0 011-1h2a1 1 0 110 2H9a1 1 0 01-1-1z" />
              <path d="M6 3a2 2 0 00-2 2v11a2 2 0 002 2h8a2 2 0 002-2V5a2 2 0 00-2-2 3 3 0 01-3 3H9a3 3 0 01-3-3z" />
            </svg>
          )}
        </button>
        
        {/* Copied notification */}
        {copied && (
          <div className="absolute top-3 right-16 bg-[#3FB950] text-white px-3 py-1 rounded-md text-sm font-medium animate-pulse">
            Copied!
          </div>
        )}
      </div>
    </div>
  );
}