#!/usr/bin/env python3

"""
Basic AHOT Tokenizer Demo

This example demonstrates basic usage of the AHOT tokenizer
with different optimization levels.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ahot import AHOTFactory, HardwareAnalyzer

def main():
    """Run basic AHOT tokenizer demo."""
    print("=== AHOT Tokenizer Basic Demo ===\n")
    
    # Analyze hardware
    print("1. Hardware Analysis")
    analyzer = HardwareAnalyzer()
    profile = analyzer.get_profile()
    analyzer.print_profile()
    print()
    
    # Test different optimization levels
    optimization_levels = ['speed', 'compression', 'balanced', 'accuracy']
    test_text = "The Adaptive Hardware-Oriented Tokenizer (AHOT) is designed to optimize performance based on your specific hardware configuration."
    
    print("2. Tokenization with Different Optimization Levels")
    print(f"Test text: {test_text[:50]}...")
    print()
    
    results = {}
    
    for level in optimization_levels:
        print(f"Testing {level} optimization:")
        
        try:
            # Create tokenizer
            tokenizer = AHOTFactory.create_optimized_tokenizer(level)
            
            # Enable monitoring
            tokenizer.enable_monitoring()
            
            # Encode text
            tokens, info = tokenizer.encode(test_text)
            
            # Get performance metrics
            metrics = tokenizer.get_monitoring_metrics()
            
            # Store results
            results[level] = {
                'encoding_time_ms': info['performance']['encoding_time_ms'],
                'compression_ratio': info['performance']['compression_ratio'],
                'throughput_tokens_per_sec': info['performance']['throughput_tokens_per_sec'],
                'output_length': tokens.shape[1]
            }
            
            print(f"  - Encoding time: {results[level]['encoding_time_ms']:.2f} ms")
            print(f"  - Compression ratio: {results[level]['compression_ratio']:.2f}x")
            print(f"  - Throughput: {results[level]['throughput_tokens_per_sec']:.0f} tokens/sec")
            print(f"  - Output length: {results[level]['output_length']} tokens")
            print()
            
        except Exception as e:
            print(f"  - Error: {e}")
            print()
    
    # Print summary
    print("3. Performance Summary")
    print("-" * 50)
    print(f"{'Level':<12} {'Time (ms)':<10} {'Compression':<12} {'Throughput':<12}")
    print("-" * 50)
    
    for level in optimization_levels:
        if level in results:
            r = results[level]
            print(f"{level:<12} {r['encoding_time_ms']:<10.2f} {r['compression_ratio']:<12.2f} {r['throughput_tokens_per_sec']:<12.0f}")
    
    print("-" * 50)
    
    # Find best performance
    if results:
        fastest = min(results.items(), key=lambda x: x[1]['encoding_time_ms'])
        best_compression = max(results.items(), key=lambda x: x[1]['compression_ratio'])
        
        print(f"\nBest performance:")
        print(f"  - Fastest: {fastest[0]} ({fastest[1]['encoding_time_ms']:.2f} ms)")
        print(f"  - Best compression: {best_compression[0]} ({best_compression[1]['compression_ratio']:.2f}x)")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main() 