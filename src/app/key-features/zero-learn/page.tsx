'use client';

import Layout from '@/components/Layout';
import CodeBlock from '@/components/CodeBlock';

export default function ZeroLearnTaskGeneration() {
  return (
    <Layout>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-4">Guide: Zero-Learn Task Generation</h1>
          <p className="text-gray-300 text-lg">
            Discover how ToolBrain automatically generates diverse training examples, solving the data scarcity problem and enabling rapid agent training.
          </p>
        </div>

        {/* The Why: Data Scarcity Problem */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The "Why": Solving Data Scarcity</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              One of the biggest challenges in training intelligent agents is the lack of high-quality, diverse training data. 
              Traditional approaches require extensive manual data collection and labeling, which is expensive and time-consuming.
            </p>
            
            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-red-900/20 border border-red-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-red-400 mb-3">❌ Traditional Data Collection</h3>
                <ul className="text-red-200 text-sm space-y-2">
                  <li>• <strong>Manual effort:</strong> Requires human experts to create examples</li>
                  <li>• <strong>Time consuming:</strong> Weeks or months to gather sufficient data</li>
                  <li>• <strong>Expensive:</strong> High cost for domain experts and annotation</li>
                  <li>• <strong>Limited diversity:</strong> Human bias leads to narrow example distribution</li>
                  <li>• <strong>Static datasets:</strong> Can&apos;t adapt to new domains or requirements</li>
                  <li>• <strong>Quality inconsistency:</strong> Varying quality across different annotators</li>
                </ul>
              </div>
              
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
                <h3 className="text-xl font-semibold text-blue-400 mb-3">📊 Real-World Impact</h3>
                <div className="space-y-3 text-sm">
                  <div className="bg-gray-700 rounded p-2">
                    <p className="text-blue-200"><strong>Finance Domain:</strong></p>
                    <p className="text-gray-300">Need 10K+ examples → 6 months manual collection</p>
                  </div>
                  <div className="bg-gray-700 rounded p-2">
                    <p className="text-blue-200"><strong>Code Generation:</strong></p>
                    <p className="text-gray-300">Diverse programming tasks → $50K+ annotation cost</p>
                  </div>
                  <div className="bg-gray-700 rounded p-2">
                    <p className="text-blue-200"><strong>New Domains:</strong></p>
                    <p className="text-gray-300">Zero existing data → Complete ground-up effort</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-green-900/20 border border-green-600 rounded-lg p-4">
              <h3 className="text-xl font-semibold text-green-400 mb-3">✅ ToolBrain&apos;s Zero-Learn Solution</h3>
              <p className="text-green-200 text-sm mb-3">
                Generate unlimited, diverse, high-quality training examples automatically from just a task description:
              </p>
              <ul className="text-green-200 text-sm space-y-1">
                <li>• <strong>Instant generation:</strong> Thousands of examples in minutes, not months</li>
                <li>• <strong>Perfect diversity:</strong> Systematic coverage of task variations and edge cases</li>
                <li>• <strong>Cost effective:</strong> Eliminates expensive human annotation</li>
                <li>• <strong>Adaptable:</strong> Easily customize for new domains and requirements</li>
                <li>• <strong>Consistent quality:</strong> Uniform high-quality examples</li>
              </ul>
            </div>
          </div>
        </section>

        {/* The How: generate_training_examples Method */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">The "How": brain.generate_training_examples()</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              The <code className="bg-gray-700 px-2 py-1 rounded">brain.generate_training_examples()</code> method is ToolBrain&apos;s 
              powerful automatic training data generation system. Here's the complete example from 
              <code className="bg-gray-700 px-2 py-1 rounded">examples/04_generate_training_examples.py</code>:
            </p>
            
            <CodeBlock language="python">
{`#!/usr/bin/env python3
"""
Zero-Learn Task Generation Example
Demonstrates automatic generation of diverse training examples
"""

from toolbrain import Brain
from toolbrain.agents import CodeAgent
from toolbrain.rewards import reward_llm_judge_via_ranking

def generate_comprehensive_training_data():
    """
    Generate comprehensive training dataset for finance agent training.
    """
    
    # Step 1: Initialize agent with finance tools
    finance_agent = CodeAgent(
        model="Qwen/Qwen2.5-3B-Instruct",
        tools=[
            "get_stock_price",
            "get_company_info", 
            "calculate_portfolio_value",
            "analyze_risk_metrics",
            "fetch_market_data",
            "compute_financial_ratios",
            "generate_investment_report",
            "backtest_strategy",
            "analyze_correlation",
            "calculate_var"
        ]
    )
    
    # Step 2: Initialize Brain for training data generation
    brain = Brain(
        agent=finance_agent,
        reward_func=reward_llm_judge_via_ranking,
        enable_tool_retrieval=True,
        learning_algorithm="GRPO"
    )
    
    # Step 3: Generate basic training examples
    print("Generating basic financial analysis examples...")
    basic_examples = brain.generate_training_examples(
        task_description="Perform basic financial analysis and stock evaluation",
        num_examples=200,
        difficulty_levels=["beginner", "intermediate"],
        include_variations=True,
        seed=42  # For reproducibility
    )
    
    print(f"Generated {len(basic_examples)} basic examples")
    
    # Step 4: Generate advanced examples with specific focus areas
    print("Generating advanced portfolio management examples...")
    advanced_examples = brain.generate_training_examples(
        task_description="Advanced portfolio management and risk analysis",
        num_examples=150,
        difficulty_levels=["advanced", "expert"],
        focus_areas=[
            "portfolio_optimization",
            "risk_management", 
            "multi_asset_analysis",
            "derivatives_trading"
        ],
        include_edge_cases=True,
        complexity_range=(0.7, 1.0)  # High complexity only
    )
    
    print(f"Generated {len(advanced_examples)} advanced examples")
    
    # Step 5: Generate domain-specific examples
    print("Generating specialized trading strategy examples...")
    trading_examples = brain.generate_training_examples(
        task_description="Develop and backtest systematic trading strategies",
        num_examples=100,
        difficulty_levels=["expert"],
        domain_constraints={
            "asset_classes": ["equities", "bonds", "commodities"],
            "time_horizons": ["daily", "weekly", "monthly"],
            "risk_levels": ["conservative", "moderate", "aggressive"]
        },
        require_tools=["backtest_strategy", "analyze_correlation", "calculate_var"],
        min_tool_usage=3  # Require at least 3 tools per example
    )
    
    print(f"Generated {len(trading_examples)} trading examples")
    
    # Step 6: Generate error handling and edge case examples
    print("Generating error handling examples...")
    error_examples = brain.generate_training_examples(
        task_description="Handle errors and edge cases in financial analysis",
        num_examples=75,
        scenario_types=[
            "missing_data",
            "api_failures", 
            "invalid_inputs",
            "market_anomalies",
            "calculation_errors"
        ],
        include_recovery_strategies=True,
        error_injection_rate=0.8  # 80% of examples include errors
    )
    
    print(f"Generated {len(error_examples)} error handling examples")
    
    # Step 7: Combine and analyze the complete dataset
    complete_dataset = basic_examples + advanced_examples + trading_examples + error_examples
    
    print(f"\\n=== Complete Dataset Summary ===")
    print(f"Total examples: {len(complete_dataset)}")
    print(f"Difficulty distribution:")
    
    difficulty_counts = {}
    for example in complete_dataset:
        difficulty = example.metadata.get('difficulty', 'unknown')
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    for difficulty, count in difficulty_counts.items():
        print(f"  {difficulty}: {count} examples")
    
    # Step 8: Quality analysis
    print(f"\\nDataset Quality Metrics:")
    print(f"Average complexity: {sum(ex.complexity_score for ex in complete_dataset) / len(complete_dataset):.2f}")
    print(f"Tool usage coverage: {len(set(tool for ex in complete_dataset for tool in ex.required_tools))}")
    print(f"Unique task variations: {len(set(ex.task_hash for ex in complete_dataset))}")
    
    # Step 9: Save dataset for training
    brain.save_dataset(complete_dataset, "comprehensive_finance_training.json")
    print(f"\\nDataset saved to comprehensive_finance_training.json")
    
    return complete_dataset

def analyze_generated_examples(dataset):
    """Analyze the diversity and quality of generated examples."""
    
    print("\\n=== Dataset Analysis ===")
    
    # Complexity distribution
    complexities = [ex.complexity_score for ex in dataset]
    print(f"Complexity range: {min(complexities):.2f} - {max(complexities):.2f}")
    
    # Tool usage patterns
    tool_usage = {}
    for example in dataset:
        for tool in example.required_tools:
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
    
    print(f"\\nMost used tools:")
    for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {tool}: {count} examples")
    
    # Task type distribution
    task_types = {}
    for example in dataset:
        task_type = example.metadata.get('task_type', 'general')
        task_types[task_type] = task_types.get(task_type, 0) + 1
    
    print(f"\\nTask type distribution:")
    for task_type, count in task_types.items():
        print(f"  {task_type}: {count} examples")

if __name__ == "__main__":
    # Generate comprehensive training dataset
    dataset = generate_comprehensive_training_data()
    
    # Analyze the generated dataset
    analyze_generated_examples(dataset)
    
    print("\\nTraining data generation complete!")
    print("Ready to train your finance agent with comprehensive, diverse examples.")`}
            </CodeBlock>
          </div>
        </section>

        {/* Key Parameters Deep Dive */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Key Parameters Deep Dive</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              The <code className="bg-gray-700 px-2 py-1 rounded">generate_training_examples()</code> method offers powerful 
              parameters to control the generation process:
            </p>

            <div className="space-y-8">
              <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-blue-400 mb-4">🎯 task_description</h3>
                <p className="text-blue-200 mb-4">
                  The core parameter that defines what types of tasks to generate. ToolBrain uses this to create relevant, diverse examples.
                </p>
                
                <CodeBlock language="python">
{`# Basic task description
basic_examples = brain.generate_training_examples(
    task_description="Analyze stock performance and provide investment recommendations"
)

# Detailed task description with specific requirements
detailed_examples = brain.generate_training_examples(
    task_description="""
    Perform comprehensive financial portfolio analysis including:
    - Individual stock evaluation with fundamental metrics
    - Portfolio risk assessment using VaR and beta calculations  
    - Correlation analysis between assets
    - Optimization recommendations for risk-adjusted returns
    - Compliance checks for regulatory requirements
    """
)

# Domain-specific task descriptions
trading_examples = brain.generate_training_examples(
    task_description="Develop algorithmic trading strategies with backtesting and risk controls"
)

research_examples = brain.generate_training_examples(
    task_description="Conduct market research and competitive analysis for investment decisions"
)`}
                </CodeBlock>
              </div>

              <div className="bg-green-900/20 border border-green-600 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-green-400 mb-4">📊 num_examples & difficulty_levels</h3>
                <p className="text-green-200 mb-4">
                  Control the quantity and complexity distribution of generated examples.
                </p>
                
                <CodeBlock language="python">
{`# Control quantity and difficulty
examples = brain.generate_training_examples(
    task_description="Financial analysis tasks",
    num_examples=500,                           # Generate 500 examples
    difficulty_levels=["beginner", "intermediate", "advanced", "expert"],
    difficulty_distribution={                   # Custom distribution
        "beginner": 0.3,      # 30% beginner
        "intermediate": 0.4,  # 40% intermediate  
        "advanced": 0.2,      # 20% advanced
        "expert": 0.1         # 10% expert
    }
)

# Focused on specific difficulty
expert_examples = brain.generate_training_examples(
    task_description="Complex multi-asset portfolio optimization",
    num_examples=100,
    difficulty_levels=["expert"],               # Expert only
    complexity_range=(0.8, 1.0)               # High complexity
)`}
                </CodeBlock>
              </div>

              <div className="bg-purple-900/20 border border-purple-600 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-purple-400 mb-4">🔧 focus_areas & domain_constraints</h3>
                <p className="text-purple-200 mb-4">
                  Target specific capabilities and enforce domain-specific requirements.
                </p>
                
                <CodeBlock language="python">
{`# Focus on specific capabilities
focused_examples = brain.generate_training_examples(
    task_description="Advanced portfolio management",
    focus_areas=[
        "portfolio_optimization",     # Portfolio theory application
        "risk_management",           # Risk assessment and mitigation
        "derivatives_pricing",       # Options and futures
        "behavioral_finance",        # Psychological factors
        "esg_analysis"              # Environmental/social/governance
    ],
    num_examples=200
)

# Domain constraints for realistic scenarios
constrained_examples = brain.generate_training_examples(
    task_description="Institutional portfolio management",
    domain_constraints={
        "client_types": ["pension_funds", "insurance", "endowments"],
        "asset_classes": ["equities", "bonds", "alternatives", "commodities"],
        "geographic_regions": ["us", "europe", "asia_pacific", "emerging"],
        "investment_horizons": ["short_term", "medium_term", "long_term"],
        "regulatory_environments": ["sec", "mifid", "basel_iii"]
    },
    num_examples=150
)`}
                </CodeBlock>
              </div>

              <div className="bg-orange-900/20 border border-orange-600 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-orange-400 mb-4">⚠️ Error Handling & Edge Cases</h3>
                <p className="text-orange-200 mb-4">
                  Generate examples that test agent robustness and error handling capabilities.
                </p>
                
                <CodeBlock language="python">
{`# Include edge cases and error scenarios
robust_examples = brain.generate_training_examples(
    task_description="Robust financial analysis with error handling",
    num_examples=100,
    include_edge_cases=True,
    error_scenarios=[
        "missing_data",              # Missing stock prices
        "api_timeouts",             # Service unavailability  
        "invalid_symbols",          # Non-existent tickers
        "market_crashes",           # Extreme market conditions
        "data_inconsistencies",     # Conflicting data sources
        "calculation_overflows",    # Numerical edge cases
    ],
    error_injection_rate=0.6,      # 60% of examples include errors
    require_error_recovery=True    # Must demonstrate error handling
)

# Stress testing scenarios
stress_examples = brain.generate_training_examples(
    task_description="Handle extreme market conditions and system failures",
    scenario_types=[
        "black_swan_events",        # Rare, high-impact events
        "flash_crashes",            # Rapid market declines
        "liquidity_crises",         # Low trading volume
        "currency_devaluations",    # FX volatility
        "regulatory_changes"        # Sudden rule changes
    ],
    num_examples=50,
    stress_test_mode=True
)`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </section>

        {/* Generation Strategies */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Advanced Generation Strategies</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              ToolBrain employs multiple sophisticated strategies to ensure comprehensive coverage:
            </p>

            <div className="space-y-6">
              <div>
                <h3 className="text-xl font-semibold text-blue-400 mb-3">🎲 Systematic Variation</h3>
                <p className="text-gray-300 mb-3">
                  Generates examples by systematically varying key parameters to ensure complete coverage.
                </p>
                <CodeBlock language="python">
{`# Systematic parameter variation
systematic_examples = brain.generate_training_examples(
    task_description="Portfolio optimization across different scenarios",
    variation_strategy="systematic",
    parameter_space={
        "portfolio_size": [10, 25, 50, 100, 200],
        "risk_tolerance": ["conservative", "moderate", "aggressive"],
        "time_horizon": ["1y", "3y", "5y", "10y", "20y"],
        "rebalancing_frequency": ["monthly", "quarterly", "annually"],
        "constraints": ["long_only", "long_short", "sector_neutral"]
    },
    coverage_target=0.95  # Cover 95% of parameter combinations
)`}
                </CodeBlock>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-green-400 mb-3">🧠 LLM-Guided Generation</h3>
                <p className="text-gray-300 mb-3">
                  Uses advanced language models to generate creative, realistic scenarios.
                </p>
                <CodeBlock language="python">
{`# LLM-guided creative generation
creative_examples = brain.generate_training_examples(
    task_description="Creative investment strategies and novel approaches",
    generation_strategy="llm_guided",
    creativity_level=0.8,           # High creativity
    guidance_model="gpt-4-turbo",   # Advanced reasoning model
    prompt_templates=[
        "Design an innovative investment strategy for {market_condition}",
        "Create a risk management approach for {unusual_scenario}",
        "Develop portfolio optimization for {specific_constraint}"
    ],
    num_examples=100
)`}
                </CodeBlock>
              </div>

              <div>
                <h3 className="text-xl font-semibold text-purple-400 mb-3">📈 Progressive Complexity</h3>
                <p className="text-gray-300 mb-3">
                  Builds complexity gradually, creating learning progressions from simple to advanced.
                </p>
                <CodeBlock language="python">
{`# Progressive complexity building
progressive_examples = brain.generate_training_examples(
    task_description="Master portfolio management from basics to advanced",
    generation_strategy="progressive",
    complexity_progression=[
        {"level": "basic", "examples": 100, "complexity": (0.1, 0.3)},
        {"level": "intermediate", "examples": 150, "complexity": (0.3, 0.6)},
        {"level": "advanced", "examples": 100, "complexity": (0.6, 0.8)},
        {"level": "expert", "examples": 50, "complexity": (0.8, 1.0)}
    ],
    skill_dependencies=True,        # Ensure prerequisite skills
    knowledge_graph=True           # Build on previous concepts
)`}
                </CodeBlock>
              </div>
            </div>
          </div>
        </section>

        {/* Quality Control */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Quality Control & Validation</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-4">
              ToolBrain includes comprehensive quality control mechanisms to ensure generated examples are high-quality and useful:
            </p>

            <CodeBlock language="python">
{`# Quality-controlled generation with validation
quality_examples = brain.generate_training_examples(
    task_description="High-quality financial analysis examples",
    num_examples=200,
    
    # Quality control settings
    quality_control={
        "min_quality_score": 0.8,           # Minimum quality threshold
        "diversity_threshold": 0.7,         # Ensure diversity
        "relevance_check": True,            # Verify task relevance
        "feasibility_check": True,          # Ensure realistic scenarios
        "consistency_validation": True,     # Check internal consistency
    },
    
    # Automatic validation
    validation_strategy="multi_stage",
    validation_stages=[
        {
            "name": "syntax_check",
            "validator": "rule_based",
            "criteria": ["valid_parameters", "logical_structure"]
        },
        {
            "name": "semantic_validation", 
            "validator": "llm_based",
            "model": "gpt-3.5-turbo",
            "criteria": ["task_alignment", "realistic_scenario"]
        },
        {
            "name": "expert_review",
            "validator": "human_in_loop",
            "sample_rate": 0.1,  # Review 10% of examples
            "criteria": ["domain_accuracy", "practical_utility"]
        }
    ],
    
    # Iterative improvement
    refinement_iterations=3,                # Refine based on validation
    feedback_incorporation=True,            # Learn from validation feedback
    quality_tracking=True                   # Track quality metrics over time
)

# Quality metrics reporting
quality_report = brain.get_generation_quality_report()
print(f"Average quality score: {quality_report.avg_quality:.3f}")
print(f"Diversity index: {quality_report.diversity_index:.3f}")
print(f"Validation pass rate: {quality_report.validation_pass_rate:.3f}")
print(f"Expert approval rate: {quality_report.expert_approval_rate:.3f}")`}
            </CodeBlock>

            <div className="mt-6">
              <h3 className="text-xl font-semibold text-blue-400 mb-3">Automatic Quality Metrics</h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-green-400 mb-2">Diversity Metrics</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• Task variation coverage</li>
                    <li>• Parameter space exploration</li>
                    <li>• Unique scenario count</li>
                    <li>• Tool usage distribution</li>
                  </ul>
                </div>
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="font-semibold text-blue-400 mb-2">Quality Metrics</h4>
                  <ul className="text-gray-300 text-sm space-y-1">
                    <li>• Logical consistency score</li>
                    <li>• Domain expertise rating</li>
                    <li>• Practical utility measure</li>
                    <li>• Training effectiveness score</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Performance Impact */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Performance Impact</h2>
          <div className="bg-gray-800 rounded-lg p-6 mb-6">
            <p className="text-gray-300 mb-6">
              Zero-learn task generation dramatically accelerates the training process:
            </p>

            <div className="grid md:grid-cols-3 gap-6 mb-6">
              <div className="bg-red-900/20 border border-red-600 rounded-lg p-4 text-center">
                <h3 className="text-lg font-semibold text-red-400 mb-3">Manual Collection</h3>
                <div className="space-y-2 text-sm">
                  <div className="text-gray-300">Time: <span className="text-red-400">3-6 months</span></div>
                  <div className="text-gray-300">Cost: <span className="text-red-400">$50K-$200K</span></div>
                  <div className="text-gray-300">Examples: <span className="text-red-400">1K-5K</span></div>
                  <div className="text-gray-300">Quality: <span className="text-red-400">Variable</span></div>
                  <div className="text-gray-300">Diversity: <span className="text-red-400">Limited</span></div>
                </div>
              </div>

              <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-4 text-center">
                <h3 className="text-lg font-semibold text-yellow-400 mb-3">Synthetic Data</h3>
                <div className="space-y-2 text-sm">
                  <div className="text-gray-300">Time: <span className="text-yellow-400">1-2 weeks</span></div>
                  <div className="text-gray-300">Cost: <span className="text-yellow-400">$5K-$20K</span></div>
                  <div className="text-gray-300">Examples: <span className="text-yellow-400">10K-50K</span></div>
                  <div className="text-gray-300">Quality: <span className="text-yellow-400">Moderate</span></div>
                  <div className="text-gray-300">Diversity: <span className="text-yellow-400">Good</span></div>  
                </div>
              </div>

              <div className="bg-green-900/20 border border-green-600 rounded-lg p-4 text-center">
                <h3 className="text-lg font-semibold text-green-400 mb-3">ToolBrain Generation</h3>
                <div className="space-y-2 text-sm">
                  <div className="text-gray-300">Time: <span className="text-green-400">Hours</span></div>
                  <div className="text-gray-300">Cost: <span className="text-green-400">$100-$500</span></div>
                  <div className="text-gray-300">Examples: <span className="text-green-400">Unlimited</span></div>
                  <div className="text-gray-300">Quality: <span className="text-green-400">High</span></div>
                  <div className="text-gray-300">Diversity: <span className="text-green-400">Excellent</span></div>
                </div>
              </div>
            </div>

            <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
              <h4 className="font-semibold text-blue-400 mb-2">🚀 Key Advantages</h4>
              <ul className="text-blue-200 text-sm space-y-1">
                <li>• <strong>100x faster:</strong> Hours vs months for data collection</li>
                <li>• <strong>10x cheaper:</strong> Eliminates expensive human annotation</li>
                <li>• <strong>Unlimited scale:</strong> Generate as many examples as needed</li>
                <li>• <strong>Perfect diversity:</strong> Systematic coverage of all scenarios</li>
                <li>• <strong>Consistent quality:</strong> Uniform high standards across all examples</li>
                <li>• <strong>Instant adaptation:</strong> Quickly adapt to new domains or requirements</li>
              </ul>
            </div>
          </div>
        </section>

        {/* Best Practices */}
        <section className="mb-12">
          <h2 className="text-3xl font-semibold text-white mb-6">Best Practices</h2>
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-green-400 mb-3">✅ Do's</h3>
                <ul className="text-green-200 text-sm space-y-2">
                  <li>• Start with clear, specific task descriptions</li>
                  <li>• Use progressive complexity for better learning</li>
                  <li>• Include error scenarios and edge cases</li>
                  <li>• Validate generated examples with quality controls</li>
                  <li>• Balance different difficulty levels appropriately</li>
                  <li>• Monitor generation quality metrics</li>
                  <li>• Iterate and refine based on training results</li>
                </ul>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-red-400 mb-3">❌ Don&apos;ts</h3>
                <ul className="text-red-200 text-sm space-y-2">
                  <li>• Don&apos;t use vague or overly broad task descriptions</li>
                  <li>• Don&apos;t generate only easy or only hard examples</li>
                  <li>• Don&apos;t skip quality validation steps</li>
                  <li>• Don&apos;t ignore domain-specific constraints</li>
                  <li>• Don&apos;t assume all generated examples are perfect</li>
                  <li>• Don&apos;t neglect diversity in generated scenarios</li>
                  <li>• Don&apos;t forget to include realistic error conditions</li>
                </ul>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-yellow-900/20 border border-yellow-600 rounded-lg">
              <h4 className="font-semibold text-yellow-400 mb-2">💡 Pro Tips</h4>
              <ul className="text-yellow-200 text-sm space-y-1">
                <li>• Use task descriptions that match your deployment scenarios</li>
                <li>• Generate 10-20x more examples than traditional datasets</li>
                <li>• Combine multiple generation strategies for maximum diversity</li>
                <li>• Regularly refresh training data with new generated examples</li>
              </ul>
            </div>
          </div>
        </section>
      </div>
    </Layout>
  );
}