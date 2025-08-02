#!/usr/bin/env python3

import torch
import time
import json
import psutil
import platform
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import logging

from ahot import AHOTFactory, HardwareAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    test_name: str
    input_length: int
    output_length: int
    encoding_time_ms: float
    decoding_time_ms: float
    compression_ratio: float
    throughput_tokens_per_sec: float
    memory_usage_mb: float
    gpu_memory_usage_mb: Optional[float]
    hardware_utilization: Dict[str, float]
    optimization_level: str
    device_type: str
    timestamp: float

@dataclass
class CrossPlatformResult:
    """Cross-platform benchmark results."""
    platform: str
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: Optional[float]
    os_type: str
    device_type: str
    processing_power: float
    benchmark_results: List[BenchmarkResult]

class AHOTBenchmarker:
    """Comprehensive benchmarking framework for AHOT tokenizer."""
    
    def __init__(self, output_dir: str = "Results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test datasets
        self.test_texts = self._load_test_datasets()
        
        # Optimization levels to test
        self.optimization_levels = ['speed', 'compression', 'balanced', 'accuracy']
        
        # Hardware analyzer
        self.hardware_analyzer = HardwareAnalyzer()
        
        logger.info("AHOT Benchmarker initialized")
    
    def _load_test_datasets(self) -> Dict[str, str]:
        """Load various test datasets for comprehensive benchmarking."""
        return {
            'short_text': "Hello, this is a short test text for AHOT tokenizer.",
            'medium_text': "This is a medium-length text that contains multiple sentences. "
                          "It includes various punctuation marks and different word lengths. "
                          "The purpose is to test the tokenizer's performance on realistic content.",
            'long_text': "This is a longer text designed to thoroughly test the AHOT tokenizer's "
                        "capabilities across different scenarios. It contains multiple paragraphs, "
                        "various sentence structures, and diverse vocabulary. The text includes "
                        "technical terms, common words, and punctuation to simulate real-world usage. "
                        "We want to ensure the tokenizer performs well across different text types "
                        "and lengths. This comprehensive testing approach helps identify potential "
                        "issues and optimization opportunities.",
            'technical_text': "The Adaptive Hardware-Oriented Tokenizer (AHOT) implements "
                             "hardware-aware optimization strategies including GPU acceleration, "
                             "mixed precision computation, and dynamic chunking algorithms. "
                             "Performance metrics include encoding time, compression ratio, "
                             "and throughput measurements across different hardware configurations.",
            'repetitive_text': "The quick brown fox jumps over the lazy dog. " * 10,
            'unicode_text': "Hello 世界! This text contains Unicode characters: éñüñöñ. "
                           "Testing internationalization support and character encoding.",
            'code_text': "def tokenize_text(text: str) -> List[int]:\n"
                        "    return [ord(c) for c in text]\n\n"
                        "class Tokenizer:\n"
                        "    def __init__(self):\n"
                        "        self.vocab_size = 50000\n"
                        "        self.embedding_dim = 256"
        }
    
    def run_single_benchmark(self, text: str, optimization_level: str = 'balanced', 
                           iterations: int = 5) -> BenchmarkResult:
        """Run a single benchmark test."""
        logger.info(f"Running benchmark: {optimization_level} optimization, {len(text)} chars")
        
        # Create tokenizer with specified optimization
        tokenizer = AHOTFactory.create_optimized_tokenizer(optimization_level)
        
        # Enable monitoring
        tokenizer.enable_monitoring()
        
        # Warm-up run
        for _ in range(2):
            _ = tokenizer.encode(text)
        
        # Benchmark runs
        encoding_times = []
        decoding_times = []
        compression_ratios = []
        memory_usages = []
        gpu_memory_usages = []
        
        for _ in range(iterations):
            # Memory measurement before
            memory_before = psutil.virtual_memory().used / (1024 * 1024)
            gpu_memory_before = None
            if torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Encoding
            start_time = time.time()
            tokens, info = tokenizer.encode(text)
            encoding_time = (time.time() - start_time) * 1000
            
            # Decoding
            start_time = time.time()
            _ = tokenizer.decode(tokens)
            decoding_time = (time.time() - start_time) * 1000
            
            # Memory measurement after
            memory_after = psutil.virtual_memory().used / (1024 * 1024)
            gpu_memory_after = None
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Store metrics
            encoding_times.append(encoding_time)
            decoding_times.append(decoding_time)
            compression_ratios.append(info['performance']['compression_ratio'])
            memory_usages.append(memory_after - memory_before)
            if gpu_memory_before is not None and gpu_memory_after is not None:
                gpu_memory_usages.append(gpu_memory_after - gpu_memory_before)
        
        # Calculate averages
        avg_encoding_time = np.mean(encoding_times)
        avg_decoding_time = np.mean(decoding_times)
        avg_compression_ratio = np.mean(compression_ratios)
        avg_memory_usage = np.mean(memory_usages)
        avg_gpu_memory_usage = np.mean(gpu_memory_usages) if gpu_memory_usages else None
        
        # Get hardware utilization
        metrics = tokenizer.get_monitoring_metrics()
        hardware_utilization = metrics.hardware_utilization if metrics else {}
        
        # Calculate throughput
        throughput = len(text) / (avg_encoding_time / 1000) if avg_encoding_time > 0 else 0
        
        # Get hardware profile
        hardware_profile = self.hardware_analyzer.get_profile()
        
        result = BenchmarkResult(
            test_name=f"{optimization_level}_optimization",
            input_length=len(text),
            output_length=tokens.shape[1],
            encoding_time_ms=avg_encoding_time,
            decoding_time_ms=avg_decoding_time,
            compression_ratio=avg_compression_ratio,
            throughput_tokens_per_sec=throughput,
            memory_usage_mb=avg_memory_usage,
            gpu_memory_usage_mb=avg_gpu_memory_usage,
            hardware_utilization=hardware_utilization,
            optimization_level=optimization_level,
            device_type=hardware_profile.device_type,
            timestamp=time.time()
        )
        
        # Disable monitoring
        tokenizer.disable_monitoring()
        
        return result
    
    def run_comprehensive_benchmark(self) -> List[BenchmarkResult]:
        """Run comprehensive benchmarks across all test cases and optimization levels."""
        logger.info("Starting comprehensive benchmark")
        
        results = []
        
        for test_name, text in self.test_texts.items():
            logger.info(f"Testing: {test_name}")
            
            for optimization_level in self.optimization_levels:
                try:
                    result = self.run_single_benchmark(text, optimization_level)
                    result.test_name = f"{test_name}_{optimization_level}"
                    results.append(result)
                    
                    logger.info(f"Completed: {result.test_name} - "
                              f"Encoding: {result.encoding_time_ms:.2f}ms, "
                              f"Compression: {result.compression_ratio:.2f}x")
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {test_name}_{optimization_level}: {e}")
        
        return results
    
    def run_cross_platform_benchmark(self) -> CrossPlatformResult:
        """Run cross-platform benchmark to compare performance across different systems."""
        logger.info("Starting cross-platform benchmark")
        
        # Get hardware profile
        hardware_profile = self.hardware_analyzer.get_profile()
        
        # Run benchmarks
        benchmark_results = self.run_comprehensive_benchmark()
        
        # Create cross-platform result
        cross_platform_result = CrossPlatformResult(
            platform=platform.platform(),
            cpu_cores=hardware_profile.cpu_cores,
            memory_gb=hardware_profile.memory_gb,
            gpu_available=hardware_profile.gpu_available,
            gpu_memory_gb=hardware_profile.gpu_memory_gb,
            os_type=hardware_profile.os_type,
            device_type=hardware_profile.device_type,
            processing_power=hardware_profile.processing_power,
            benchmark_results=benchmark_results
        )
        
        return cross_platform_result
    
    def analyze_results(self, results: List[BenchmarkResult]) -> Dict:
        """Analyze benchmark results and generate insights."""
        logger.info("Analyzing benchmark results")
        
        analysis = {
            'summary': {},
            'optimization_comparison': {},
            'performance_metrics': {},
            'hardware_analysis': {},
            'recommendations': {}
        }
        
        # Group results by optimization level
        by_optimization = {}
        for result in results:
            opt_level = result.optimization_level
            if opt_level not in by_optimization:
                by_optimization[opt_level] = []
            by_optimization[opt_level].append(result)
        
        # Calculate summary statistics
        for opt_level, opt_results in by_optimization.items():
            encoding_times = [r.encoding_time_ms for r in opt_results]
            compression_ratios = [r.compression_ratio for r in opt_results]
            throughputs = [r.throughput_tokens_per_sec for r in opt_results]
            
            analysis['optimization_comparison'][opt_level] = {
                'avg_encoding_time_ms': np.mean(encoding_times),
                'avg_compression_ratio': np.mean(compression_ratios),
                'avg_throughput_tokens_per_sec': np.mean(throughputs),
                'min_encoding_time_ms': np.min(encoding_times),
                'max_encoding_time_ms': np.max(encoding_times),
                'std_encoding_time_ms': np.std(encoding_times)
            }
        
        # Overall summary
        all_encoding_times = [r.encoding_time_ms for r in results]
        all_compression_ratios = [r.compression_ratio for r in results]
        all_throughputs = [r.throughput_tokens_per_sec for r in results]
        
        analysis['summary'] = {
            'total_tests': len(results),
            'avg_encoding_time_ms': np.mean(all_encoding_times),
            'avg_compression_ratio': np.mean(all_compression_ratios),
            'avg_throughput_tokens_per_sec': np.mean(all_throughputs),
            'best_optimization': min(by_optimization.keys(), 
                                   key=lambda x: np.mean([r.encoding_time_ms for r in by_optimization[x]]))
        }
        
        # Performance metrics
        analysis['performance_metrics'] = {
            'encoding_performance': {
                'fastest': min(results, key=lambda x: x.encoding_time_ms),
                'slowest': max(results, key=lambda x: x.encoding_time_ms),
                'best_compression': max(results, key=lambda x: x.compression_ratio),
                'highest_throughput': max(results, key=lambda x: x.throughput_tokens_per_sec)
            }
        }
        
        # Hardware analysis
        gpu_results = [r for r in results if r.gpu_memory_usage_mb is not None]
        cpu_results = [r for r in results if r.gpu_memory_usage_mb is None]
        
        analysis['hardware_analysis'] = {
            'gpu_enabled_tests': len(gpu_results),
            'cpu_only_tests': len(cpu_results),
            'avg_gpu_memory_usage_mb': np.mean([r.gpu_memory_usage_mb for r in gpu_results]) if gpu_results else None,
            'avg_cpu_memory_usage_mb': np.mean([r.memory_usage_mb for r in cpu_results]) if cpu_results else None
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(results)
        
        return analysis
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> Dict:
        """Generate optimization recommendations based on benchmark results."""
        recommendations = {
            'optimal_optimization': {},
            'hardware_suggestions': {},
            'performance_tips': []
        }
        
        # Find best optimization for different scenarios
        by_optimization = {}
        for result in results:
            opt_level = result.optimization_level
            if opt_level not in by_optimization:
                by_optimization[opt_level] = []
            by_optimization[opt_level].append(result)
        
        # Speed optimization
        speed_optimized = min(by_optimization.keys(), 
                            key=lambda x: np.mean([r.encoding_time_ms for r in by_optimization[x]]))
        
        # Compression optimization
        compression_optimized = max(by_optimization.keys(), 
                                 key=lambda x: np.mean([r.compression_ratio for r in by_optimization[x]]))
        
        # Balanced optimization
        balanced_optimized = 'balanced' if 'balanced' in by_optimization else speed_optimized
        
        recommendations['optimal_optimization'] = {
            'speed': speed_optimized,
            'compression': compression_optimized,
            'balanced': balanced_optimized
        }
        
        # Hardware suggestions
        hardware_profile = self.hardware_analyzer.get_profile()
        
        if hardware_profile.gpu_available:
            recommendations['hardware_suggestions']['gpu'] = "GPU acceleration is available and recommended"
        else:
            recommendations['hardware_suggestions']['gpu'] = "Consider using GPU for better performance"
        
        if hardware_profile.memory_gb < 8:
            recommendations['hardware_suggestions']['memory'] = "Consider increasing memory for better performance"
        else:
            recommendations['hardware_suggestions']['memory'] = "Memory is sufficient for optimal performance"
        
        # Performance tips
        recommendations['performance_tips'] = [
            "Use 'speed' optimization for real-time applications",
            "Use 'compression' optimization for storage-constrained scenarios",
            "Use 'balanced' optimization for general-purpose applications",
            "Enable GPU acceleration when available for better performance",
            "Monitor memory usage for large text inputs"
        ]
        
        return recommendations
    
    def save_results(self, results: List[BenchmarkResult], analysis: Dict, 
                    filename: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        output_file = self.output_dir / filename
        
        # Convert dataclass objects to dictionaries and handle numpy types
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # Handle dataclass objects
                return convert_to_serializable(asdict(obj))
            elif hasattr(obj, 'item'):
                # Handle numpy scalars
                return obj.item()
            elif isinstance(obj, (np.integer, np.floating)):
                # Handle numpy numeric types
                return obj.item()
            else:
                return obj
        
        data = {
            'timestamp': time.time(),
            'hardware_profile': convert_to_serializable(self.hardware_analyzer.get_profile()),
            'results': [convert_to_serializable(result) for result in results],
            'analysis': convert_to_serializable(analysis)
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    def generate_visualizations(self, results: List[BenchmarkResult], analysis: Dict):
        """Generate visualization plots for benchmark results."""
        logger.info("Generating visualizations")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AHOT Tokenizer Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Encoding time comparison by optimization level
        ax1 = axes[0, 0]
        optimization_levels = list(analysis['optimization_comparison'].keys())
        avg_encoding_times = [analysis['optimization_comparison'][opt]['avg_encoding_time_ms'] 
                             for opt in optimization_levels]
        
        bars1 = ax1.bar(optimization_levels, avg_encoding_times, color='skyblue', alpha=0.7)
        ax1.set_title('Average Encoding Time by Optimization Level')
        ax1.set_ylabel('Encoding Time (ms)')
        ax1.set_xlabel('Optimization Level')
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_encoding_times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # 2. Compression ratio comparison
        ax2 = axes[0, 1]
        avg_compression_ratios = [analysis['optimization_comparison'][opt]['avg_compression_ratio'] 
                                 for opt in optimization_levels]
        
        bars2 = ax2.bar(optimization_levels, avg_compression_ratios, color='lightgreen', alpha=0.7)
        ax2.set_title('Average Compression Ratio by Optimization Level')
        ax2.set_ylabel('Compression Ratio')
        ax2.set_xlabel('Optimization Level')
        
        for bar, value in zip(bars2, avg_compression_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Throughput comparison
        ax3 = axes[1, 0]
        avg_throughputs = [analysis['optimization_comparison'][opt]['avg_throughput_tokens_per_sec'] 
                          for opt in optimization_levels]
        
        bars3 = ax3.bar(optimization_levels, avg_throughputs, color='salmon', alpha=0.7)
        ax3.set_title('Average Throughput by Optimization Level')
        ax3.set_ylabel('Throughput (tokens/sec)')
        ax3.set_xlabel('Optimization Level')
        
        for bar, value in zip(bars3, avg_throughputs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # 4. Memory usage comparison
        ax4 = axes[1, 1]
        memory_usages = [r.memory_usage_mb for r in results]
        gpu_memory_usages = [r.gpu_memory_usage_mb for r in results if r.gpu_memory_usage_mb is not None]
        
        if gpu_memory_usages:
            ax4.hist(memory_usages, alpha=0.7, label='CPU Memory', bins=10, color='lightblue')
            ax4.hist(gpu_memory_usages, alpha=0.7, label='GPU Memory', bins=10, color='orange')
            ax4.legend()
        else:
            ax4.hist(memory_usages, alpha=0.7, bins=10, color='lightblue')
        
        ax4.set_title('Memory Usage Distribution')
        ax4.set_ylabel('Frequency')
        ax4.set_xlabel('Memory Usage (MB)')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "benchmark_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {plot_file}")
    
    def run_full_benchmark_suite(self):
        """Run the complete benchmark suite with analysis and visualization."""
        logger.info("Starting full benchmark suite")
        
        # Run comprehensive benchmark
        results = self.run_comprehensive_benchmark()
        
        # Analyze results
        analysis = self.analyze_results(results)
        
        # Save results
        self.save_results(results, analysis)
        
        # Generate visualizations
        self.generate_visualizations(results, analysis)
        
        # Print summary
        self._print_benchmark_summary(analysis)
        
        logger.info("Benchmark suite completed")
    
    def _print_benchmark_summary(self, analysis: Dict):
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("AHOT TOKENIZER BENCHMARK SUMMARY")
        print("="*60)
        
        print(f"\nTotal Tests: {analysis['summary']['total_tests']}")
        print(f"Average Encoding Time: {analysis['summary']['avg_encoding_time_ms']:.2f} ms")
        print(f"Average Compression Ratio: {analysis['summary']['avg_compression_ratio']:.2f}x")
        print(f"Average Throughput: {analysis['summary']['avg_throughput_tokens_per_sec']:.0f} tokens/sec")
        print(f"Best Optimization Level: {analysis['summary']['best_optimization']}")
        
        print("\nOptimization Level Performance:")
        for opt_level, metrics in analysis['optimization_comparison'].items():
            print(f"  {opt_level.upper()}:")
            print(f"    Encoding Time: {metrics['avg_encoding_time_ms']:.2f} ms")
            print(f"    Compression Ratio: {metrics['avg_compression_ratio']:.2f}x")
            print(f"    Throughput: {metrics['avg_throughput_tokens_per_sec']:.0f} tokens/sec")
        
        print("\nRecommendations:")
        for category, recs in analysis['recommendations'].items():
            if isinstance(recs, dict):
                print(f"  {category.replace('_', ' ').title()}:")
                for key, value in recs.items():
                    print(f"    {key}: {value}")
            elif isinstance(recs, list):
                print(f"  {category.replace('_', ' ').title()}:")
                for rec in recs:
                    print(f"    • {rec}")
        
        print("="*60)

if __name__ == "__main__":
    # Create benchmarker
    benchmarker = AHOTBenchmarker()
    
    # Run full benchmark suite
    benchmarker.run_full_benchmark_suite() 