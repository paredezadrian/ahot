# AHOT API Reference

This document outlines the primary classes and functions available in the
Adaptive Hardware-Oriented Tokenizer (AHOT).

## HardwareAnalyzer
Detects system capabilities and produces a `HardwareProfile` describing CPU
cores, memory, GPU availability and other metrics.

### Methods
- `get_profile() -> HardwareProfile`: return the collected profile.
- `print_profile()`: print a human readable summary of the profile.

## AHOTFactory
Factory helpers for building tokenizers that suit the current hardware.

### Methods
- `create_tokenizer(vocab_size=50000, embedding_dim=256) -> AHOTTokenizer`:
  build a basic tokenizer using the detected hardware profile.
- `create_optimized_tokenizer(optimization_level='balanced') -> OptimizedAHOTTokenizer`:
  build a tokenizer configured for `speed`, `compression`, `balanced` or
  `accuracy` optimisation strategies.

## AHOTTokenizer
Core tokenizer implementation that adapts its architecture to the hardware
profile. Provides encode/decode capabilities and optional performance
monitoring.

### Methods
- `encode(text: str) -> Tuple[Tensor, Dict]`: convert text to token tensor
  and return performance information.
- `decode(tokens: Tensor) -> str`: reconstruct text from token tensor.
- `enable_monitoring(start_monitoring: bool = True)`: start collecting
  runtime metrics.
- `disable_monitoring()`: stop collecting metrics.
- `get_monitoring_metrics() -> Optional[PerformanceMetrics]`: retrieve the
  latest metrics snapshot.

## OptimizedAHOTTokenizer
Subclass of `AHOTTokenizer` that applies a specific optimisation strategy such
as speed or compression. Optimisations influence layer counts, hidden
dimensions and GPU usage.

## ProductionMonitor
Utility used by the tokenizer to record encoding time, memory usage, cache hit
rates and other runtime metrics.

### Key Methods
- `start_monitoring()` / `stop_monitoring()`
- `record_encoding_time(time_ms: float)`
- `record_compression_ratio(ratio: float)`
- `get_current_metrics() -> PerformanceMetrics`
- `export_metrics(filepath: str)`

## AHOTBenchmarker
High level benchmarking utilities for evaluating tokenizer performance across
multiple texts and optimisation strategies.

### Methods
- `run_single_benchmark(text: str, optimization_level: str) -> BenchmarkResult`
- `run_comprehensive_benchmark() -> List[BenchmarkResult]`
- `run_full_benchmark_suite()`: run the full suite and generate visualisations.

