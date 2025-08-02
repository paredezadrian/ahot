#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import time
import json
from pathlib import Path
import logging

from ahot import AHOTFactory, HardwareAnalyzer, ProductionMonitor
from ahot_benchmark import AHOTBenchmarker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demonstrate_hardware_analysis():
    """Demonstrate enhanced hardware analysis capabilities."""
    print("\n" + "="*60)
    print("🔍 HARDWARE ANALYSIS DEMONSTRATION")
    print("="*60)
    
    analyzer = HardwareAnalyzer()
    profile = analyzer.get_profile()
    
    print(f"Device Type: {profile.device_type}")
    print(f"CPU Cores: {profile.cpu_cores}")
    print(f"Memory: {profile.memory_gb:.1f} GB")
    print(f"GPU Available: {profile.gpu_available}")
    if profile.gpu_memory_gb:
        print(f"GPU Memory: {profile.gpu_memory_gb:.1f} GB")
    print(f"Processing Power: {profile.processing_power:.2f}")
    print(f"Memory Efficiency: {profile.memory_efficiency:.2f}")
    if profile.cache_size_mb:
        print(f"Cache Size: {profile.cache_size_mb:.1f} MB")

def demonstrate_gpu_acceleration():
    """Demonstrate GPU acceleration capabilities."""
    print("\n" + "="*60)
    print("🚀 GPU ACCELERATION DEMONSTRATION")
    print("="*60)
    
    # Create tokenizers with different optimization levels
    optimization_levels = ['speed', 'balanced', 'compression', 'accuracy']
    
    test_text = "This is a comprehensive test of the AHOT tokenizer with GPU acceleration capabilities."
    
    for opt_level in optimization_levels:
        print(f"\nTesting {opt_level.upper()} optimization:")
        
        # Create optimized tokenizer
        tokenizer = AHOTFactory.create_optimized_tokenizer(opt_level)
        
        # Get hardware optimizations
        optimizations = tokenizer.get_hardware_optimizations()
        
        print(f"  GPU Acceleration: {optimizations.get('gpu_acceleration', False)}")
        print(f"  Optimization Level: {optimizations.get('optimization_level', 'none')}")
        print(f"  Half Precision: {optimizations.get('use_half_precision', False)}")
        print(f"  Chunk Size: {optimizations.get('chunk_size', 'N/A')}")
        print(f"  Compression Factor: {optimizations.get('compression_factor', 'N/A')}")
        
        # Test encoding
        start_time = time.time()
        tokens, info = tokenizer.encode(test_text)
        encoding_time = (time.time() - start_time) * 1000
        
        print(f"  Encoding Time: {encoding_time:.2f} ms")
        print(f"  Compression Ratio: {info['performance']['compression_ratio']:.2f}x")
        print(f"  Throughput: {info['performance']['throughput_tokens_per_sec']:.0f} tokens/sec")

def demonstrate_production_monitoring():
    """Demonstrate production monitoring capabilities."""
    print("\n" + "="*60)
    print("📊 PRODUCTION MONITORING DEMONSTRATION")
    print("="*60)
    
    # Create tokenizer with monitoring enabled
    tokenizer = AHOTFactory.create_optimized_tokenizer('balanced')
    tokenizer.enable_monitoring()
    
    # Test texts of varying lengths
    test_texts = [
        "Short text.",
        "This is a medium-length text that will be processed by the AHOT tokenizer to demonstrate monitoring capabilities.",
        "This is a much longer text designed to thoroughly test the production monitoring system. " * 5
    ]
    
    print("Running monitored encoding tests...")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {len(text)} characters")
        
        # Encode with monitoring
        tokens, info = tokenizer.encode(text)
        
        # Get current metrics
        metrics = tokenizer.get_monitoring_metrics()
        
        if metrics:
            print(f"  Encoding Time: {metrics.encoding_time_ms:.2f} ms")
            print(f"  Compression Ratio: {metrics.compression_ratio:.2f}x")
            print(f"  Throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/sec")
            print(f"  Memory Usage: {metrics.memory_usage_mb:.1f} MB")
            if metrics.gpu_memory_usage_mb:
                print(f"  GPU Memory Usage: {metrics.gpu_memory_usage_mb:.1f} MB")
            print(f"  CPU Utilization: {metrics.hardware_utilization.get('cpu_percent', 0):.1f}%")
            print(f"  Memory Utilization: {metrics.hardware_utilization.get('memory_percent', 0):.1f}%")
            if metrics.hardware_utilization.get('gpu_percent'):
                print(f"  GPU Utilization: {metrics.hardware_utilization['gpu_percent']:.1f}%")
    
    # Export monitoring data
    output_file = "Results/production_monitoring_demo.json"
    tokenizer.export_monitoring_data(output_file)
    print(f"\nMonitoring data exported to {output_file}")
    
    # Disable monitoring
    tokenizer.disable_monitoring()

def demonstrate_multi_platform_testing():
    """Demonstrate multi-platform testing capabilities."""
    print("\n" + "="*60)
    print("🖥️ MULTI-PLATFORM TESTING DEMONSTRATION")
    print("="*60)
    
    # Create benchmarker
    benchmarker = AHOTBenchmarker()
    
    print("Running comprehensive benchmark suite...")
    print("This will test all optimization levels across different text types.")
    
    # Run a subset of benchmarks for demonstration
    results = []
    
    # Test with different text types and optimization levels
    test_cases = [
        ("short_text", "speed"),
        ("medium_text", "balanced"),
        ("technical_text", "compression"),
        ("long_text", "accuracy")
    ]
    
    for test_name, opt_level in test_cases:
        print(f"\nTesting {test_name} with {opt_level} optimization...")
        
        text = benchmarker.test_texts[test_name]
        result = benchmarker.run_single_benchmark(text, opt_level)
        results.append(result)
        
        print(f"  Encoding Time: {result.encoding_time_ms:.2f} ms")
        print(f"  Compression Ratio: {result.compression_ratio:.2f}x")
        print(f"  Throughput: {result.throughput_tokens_per_sec:.0f} tokens/sec")
        print(f"  Memory Usage: {result.memory_usage_mb:.1f} MB")
        if result.gpu_memory_usage_mb:
            print(f"  GPU Memory Usage: {result.gpu_memory_usage_mb:.1f} MB")
    
    # Analyze results
    analysis = benchmarker.analyze_results(results)
    
    print("\nBenchmark Analysis Summary:")
    print(f"  Total Tests: {analysis['summary']['total_tests']}")
    print(f"  Average Encoding Time: {analysis['summary']['avg_encoding_time_ms']:.2f} ms")
    print(f"  Average Compression Ratio: {analysis['summary']['avg_compression_ratio']:.2f}x")
    print(f"  Best Optimization: {analysis['summary']['best_optimization']}")
    
    # Save results
    benchmarker.save_results(results, analysis, "multi_platform_demo_results.json")
    print("\nResults saved to Results/multi_platform_demo_results.json")
    
    # Generate visualizations
    benchmarker.generate_visualizations(results, analysis)
    print("Multi-platform visualizations saved to Results/benchmark_analysis.png")

def demonstrate_cross_platform_benchmarking():
    """Demonstrate cross-platform benchmarking."""
    print("\n" + "="*60)
    print("🌐 CROSS-PLATFORM BENCHMARKING DEMONSTRATION")
    print("="*60)
    
    # Create benchmarker
    benchmarker = AHOTBenchmarker()
    
    print("Running cross-platform benchmark...")
    
    # Run cross-platform benchmark
    cross_platform_result = benchmarker.run_cross_platform_benchmark()
    
    print(f"\nPlatform: {cross_platform_result.platform}")
    print(f"Device Type: {cross_platform_result.device_type}")
    print(f"CPU Cores: {cross_platform_result.cpu_cores}")
    print(f"Memory: {cross_platform_result.memory_gb:.1f} GB")
    print(f"GPU Available: {cross_platform_result.gpu_available}")
    if cross_platform_result.gpu_memory_gb:
        print(f"GPU Memory: {cross_platform_result.gpu_memory_gb:.1f} GB")
    print(f"Processing Power: {cross_platform_result.processing_power:.2f}")
    print(f"OS Type: {cross_platform_result.os_type}")
    
    print(f"\nTotal Benchmark Results: {len(cross_platform_result.benchmark_results)}")
    
    # Analyze cross-platform results
    analysis = benchmarker.analyze_results(cross_platform_result.benchmark_results)
    
    print("\nCross-Platform Performance:")
    for opt_level, metrics in analysis['optimization_comparison'].items():
        print(f"  {opt_level.upper()}:")
        print(f"    Avg Encoding Time: {metrics['avg_encoding_time_ms']:.2f} ms")
        print(f"    Avg Compression Ratio: {metrics['avg_compression_ratio']:.2f}x")
        print(f"    Avg Throughput: {metrics['avg_throughput_tokens_per_sec']:.0f} tokens/sec")
    
    # Save cross-platform results
    benchmarker.save_results(
        cross_platform_result.benchmark_results, 
        analysis, 
        "cross_platform_benchmark_results.json"
    )
    print("\nCross-platform results saved to Results/cross_platform_benchmark_results.json")
    
    # Generate visualizations
    benchmarker.generate_visualizations(
        cross_platform_result.benchmark_results, 
        analysis
    )
    print("Cross-platform visualizations saved to Results/benchmark_analysis.png")

def demonstrate_real_time_monitoring():
    """Demonstrate real-time monitoring capabilities."""
    print("\n" + "="*60)
    print("⏱️ REAL-TIME MONITORING DEMONSTRATION")
    print("="*60)
    
    # Create tokenizer with monitoring
    tokenizer = AHOTFactory.create_optimized_tokenizer('balanced')
    tokenizer.enable_monitoring()
    
    # Simulate real-time processing
    test_text = "This is a test text for real-time monitoring demonstration. " * 10
    
    print("Starting real-time monitoring...")
    print("Processing text in real-time with live metrics...")
    
    for i in range(5):
        print(f"\nIteration {i + 1}:")
        
        # Process text
        tokens, info = tokenizer.encode(test_text)
        
        # Get real-time metrics
        metrics = tokenizer.get_monitoring_metrics()
        
        if metrics:
            print(f"  Encoding Time: {metrics.encoding_time_ms:.2f} ms")
            print(f"  Compression Ratio: {metrics.compression_ratio:.2f}x")
            print(f"  Throughput: {metrics.throughput_tokens_per_sec:.0f} tokens/sec")
            print(f"  CPU Usage: {metrics.hardware_utilization.get('cpu_percent', 0):.1f}%")
            print(f"  Memory Usage: {metrics.memory_usage_mb:.1f} MB")
            if metrics.gpu_memory_usage_mb:
                print(f"  GPU Memory: {metrics.gpu_memory_usage_mb:.1f} MB")
        
        # Simulate processing delay
        time.sleep(1)
    
    # Get final metrics summary
    metrics_summary = tokenizer.monitor.get_metrics_summary()
    
    print("\nFinal Metrics Summary:")
    if metrics_summary:
        print(f"  Average Encoding Time: {metrics_summary.get('encoding_time', {}).get('avg', 0):.2f} ms")
        print(f"  Average Compression Ratio: {metrics_summary.get('compression_ratio', {}).get('avg', 0):.2f}x")
        print(f"  Average Throughput: {metrics_summary.get('throughput', {}).get('avg', 0):.0f} tokens/sec")
        print(f"  Average CPU Usage: {metrics_summary.get('hardware_utilization', {}).get('avg_cpu', 0):.1f}%")
        print(f"  Average Memory Usage: {metrics_summary.get('hardware_utilization', {}).get('avg_memory', 0):.1f}%")
    
    # Disable monitoring
    tokenizer.disable_monitoring()

def demonstrate_advanced_features():
    """Demonstrate advanced features and optimizations."""
    print("\n" + "="*60)
    print("🎯 ADVANCED FEATURES DEMONSTRATION")
    print("="*60)
    
    # Test different optimization strategies
    optimization_levels = ['speed', 'compression', 'balanced', 'accuracy']
    
    test_text = "Advanced demonstration of AHOT tokenizer features including GPU acceleration, "
    test_text += "mixed precision computation, and hardware-aware optimizations."
    
    print("Testing different optimization strategies:")
    
    for opt_level in optimization_levels:
        print(f"\n{opt_level.upper()} Optimization:")
        
        # Create tokenizer
        tokenizer = AHOTFactory.create_optimized_tokenizer(opt_level)
        
        # Get optimizations
        optimizations = tokenizer.get_hardware_optimizations()
        
        print(f"  Chunk Size: {optimizations.get('chunk_size', 'N/A')}")
        print(f"  Compression Factor: {optimizations.get('compression_factor', 'N/A')}")
        print(f"  Batch Limit: {optimizations.get('batch_limit', 'N/A')}")
        print(f"  Half Precision: {optimizations.get('use_half_precision', False)}")
        print(f"  GPU Acceleration: {optimizations.get('gpu_acceleration', False)}")
        print(f"  Optimization Level: {optimizations.get('optimization_level', 'none')}")
        
        # Test performance
        start_time = time.time()
        tokens, info = tokenizer.encode(test_text)
        encoding_time = (time.time() - start_time) * 1000
        
        print(f"  Encoding Time: {encoding_time:.2f} ms")
        print(f"  Compression Ratio: {info['performance']['compression_ratio']:.2f}x")
        print(f"  Throughput: {info['performance']['throughput_tokens_per_sec']:.0f} tokens/sec")
        
        # Show hardware optimizations
        hw_opt = info.get('hardware_optimizations', {})
        print(f"  Device: {hw_opt.get('device', 'N/A')}")
        print(f"  Processing Power: {hw_opt.get('processing_power', 0):.2f}")

def main():
    """Run the complete enhanced AHOT demonstration."""
    print("🚀 ENHANCED AHOT TOKENIZER DEMONSTRATION")
    print("="*60)
    print("This demonstration showcases the high-priority enhancements:")
    print("• Production Monitoring with Performance Metrics")
    print("• GPU Acceleration with CUDA Support")
    print("• Multi-Platform Testing")
    print("• Cross-Platform Benchmarking")
    print("• Real-time Hardware Monitoring")
    print("="*60)
    
    try:
        # Create Results directory if it doesn't exist
        Path("Results").mkdir(exist_ok=True)
        
        # Run demonstrations
        demonstrate_hardware_analysis()
        demonstrate_gpu_acceleration()
        demonstrate_production_monitoring()
        demonstrate_multi_platform_testing()
        demonstrate_cross_platform_benchmarking()
        demonstrate_real_time_monitoring()
        demonstrate_advanced_features()
        
        print("\n" + "="*60)
        print("✅ ENHANCED AHOT DEMONSTRATION COMPLETED")
        print("="*60)
        print("All high-priority enhancements have been successfully demonstrated!")
        print("Check the Results/ directory for generated files and visualizations.")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")

if __name__ == "__main__":
    main() 