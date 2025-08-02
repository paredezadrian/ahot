#!/usr/bin/env python3

"""
Full Benchmark Suite Runner for AHOT Tokenizer
This script runs the complete benchmark suite with analysis and visualization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ahot_benchmark import AHOTBenchmarker

def main():
    """Run the complete benchmark suite."""
    print("🚀 Running Full AHOT Benchmark Suite")
    print("=" * 50)
    
    # Create benchmarker
    benchmarker = AHOTBenchmarker()
    
    # Run full benchmark suite
    benchmarker.run_full_benchmark_suite()
    
    print("\n✅ Benchmark suite completed!")
    print("📊 Check the Results/ directory for:")
    print("   • benchmark_results.json - Detailed benchmark data")
    print("   • benchmark_analysis.png - Performance visualizations")

if __name__ == "__main__":
    main() 