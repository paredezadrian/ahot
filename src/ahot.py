#!/usr/bin/env python3
"""Core implementation for the Adaptive Hardware-Oriented Tokenizer (AHOT).

This module contains utilities for analysing system hardware, building
hardware-aware tokenizers and monitoring their runtime performance.
"""

import torch
import torch.nn as nn
import numpy as np
import psutil
import platform
import time
import subprocess
import re
import os
import shutil
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import json
import logging
import threading
import queue
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HardwareProfile:
    cpu_cores: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: Optional[float]
    os_type: str
    device_type: str
    processing_power: float
    memory_efficiency: float
    cache_size_mb: Optional[float] = None
    cpu_frequency_ghz: Optional[float] = None

@dataclass
class PerformanceMetrics:
    """Production monitoring metrics for AHOT tokenizer."""
    encoding_time_ms: float
    decoding_time_ms: float
    memory_usage_mb: float
    gpu_memory_usage_mb: Optional[float]
    compression_ratio: float
    cache_hit_rate: float
    throughput_tokens_per_sec: float
    accuracy_score: float
    hardware_utilization: Dict[str, float]
    timestamp: float

@dataclass
class CacheMetrics:
    """Cache performance monitoring."""
    hit_count: int
    miss_count: int
    total_requests: int
    hit_rate: float
    cache_size_mb: float
    eviction_count: int
    avg_access_time_ms: float

class HardwareAnalyzer:
    """Detects hardware capabilities for configuring the tokenizer."""

    def __init__(self):
        self.profile = self._analyze_hardware()
        logger.info(f"Hardware analysis completed: {self.profile.device_type}")
    
    def _analyze_hardware(self) -> HardwareProfile:
        """Collect detailed information about CPU, memory and GPU."""
        cpu_cores = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_frequency_ghz = None
        if cpu_freq and cpu_freq.current:
            cpu_frequency_ghz = cpu_freq.current / 1000.0
        
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        gpu_available = torch.cuda.is_available()
        gpu_memory_gb = None
        if gpu_available:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        os_type = platform.system()
        device_type = self._classify_device(memory_gb, cpu_cores, gpu_available)
        processing_power = self._calculate_processing_power(cpu_cores, memory_gb, gpu_available, cpu_frequency_ghz)
        memory_efficiency = self._calculate_memory_efficiency(memory_gb, os_type)
        cache_size_mb = self._analyze_cache()
        
        return HardwareProfile(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            os_type=os_type,
            device_type=device_type,
            processing_power=processing_power,
            memory_efficiency=memory_efficiency,
            cache_size_mb=cache_size_mb,
            cpu_frequency_ghz=cpu_frequency_ghz
        )
    
    def _classify_device(self, memory_gb: float, cpu_cores: int, gpu_available: bool) -> str:
        if memory_gb < 0:
            raise ValueError("memory_gb must be non-negative")
        
        if cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        
        if memory_gb >= 32 and cpu_cores >= 16:
            return 'server'
        elif memory_gb >= 16 and cpu_cores >= 8:
            return 'desktop'
        elif memory_gb >= 8 and cpu_cores >= 4:
            return 'laptop'
        else:
            return 'mobile'
    
    def _calculate_processing_power(self, cpu_cores: int, memory_gb: float,
                                  gpu_available: bool, cpu_freq_ghz: Optional[float]) -> float:
        if cpu_cores <= 0:
            raise ValueError("cpu_cores must be positive")
        
        if memory_gb < 0:
            raise ValueError("memory_gb must be non-negative")
        
        if cpu_freq_ghz is not None and cpu_freq_ghz <= 0:
            raise ValueError("cpu_freq_ghz must be positive if provided")
        
        # Normalise CPU and memory contributions to a 0-1 range
        cpu_score = min(cpu_cores / 32, 1.0)
        memory_score = min(memory_gb / 64, 1.0)
        
        freq_bonus = 0.0
        if cpu_freq_ghz:
            freq_bonus = min((cpu_freq_ghz - 2.0) / 4.0, 0.2)
        
        gpu_bonus = 0.3 if gpu_available else 0.0
        
        # Weighted sum prioritises CPU and memory while still accounting for
        # GPU presence and higher clock speeds.
        processing_power = (0.4 * cpu_score +
                          0.3 * memory_score +
                          0.2 * gpu_bonus +
                          0.1 * freq_bonus)
        
        return min(processing_power, 1.0)
    
    def _calculate_memory_efficiency(self, memory_gb: float, os_type: str) -> float:
        """Estimate how effectively the operating system uses available RAM."""
        if memory_gb < 0:
            raise ValueError("memory_gb must be non-negative")
        
        if not isinstance(os_type, str) or not os_type.strip():
            raise ValueError("os_type must be a non-empty string")
        
        base_efficiency = min(memory_gb / 32, 1.0)
        
        os_efficiency = {
            'Windows': 0.8,
            'Linux': 0.95,
            'Darwin': 0.9
        }
        
        os_factor = os_efficiency.get(os_type, 0.85)
        return base_efficiency * os_factor
    
    def _analyze_cache(self) -> Optional[float]:
        """
        Analyze CPU cache using platform-specific APIs.
        Returns total cache size in MB or None if analysis fails.
        """
        try:
            os_type = platform.system()
            
            if os_type == "Windows":
                return self._analyze_cache_windows()
            elif os_type == "Linux":
                return self._analyze_cache_linux()
            elif os_type == "Darwin":
                return self._analyze_cache_macos()
            else:
                return self._analyze_cache_generic()
                
        except Exception as e:
            logger.warning(f"Cache analysis failed: {e}")
            return None

    def _run_command(self, command: List[str]) -> Optional[subprocess.CompletedProcess]:
        """Safely run a system command and return the CompletedProcess."""
        cmd_path = shutil.which(command[0])
        if not cmd_path:
            logger.debug(f"{command[0]} command not found")
            return None
        try:
            return subprocess.run([cmd_path, *command[1:]], capture_output=True, text=True, timeout=10, shell=False)
        except Exception as e:
            logger.debug(f"Command {command[0]} failed: {e}")
            return None

    def _analyze_cache_windows(self) -> Optional[float]:
        """Analyze cache on Windows using multiple methods."""
        try:
            result = self._run_command([
                'wmic',
                'cpu',
                'get',
                'L2CacheSize,L3CacheSize',
                '/format:csv',
            ])
            if result and result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split(',')
                    if len(parts) >= 3:
                        l2_size = self._parse_cache_size(parts[1])
                        l3_size = self._parse_cache_size(parts[2])
                        total_cache = (l2_size or 0) + (l3_size or 0)
                        if total_cache > 0:
                            return total_cache / 1024.0

            result = self._run_command([
                'powershell',
                '-Command',
                'Get-WmiObject -Class Win32_Processor | Select-Object L2CacheSize,L3CacheSize | ConvertTo-Csv',
            ])
            if result and result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 3:
                    data_line = lines[2]
                    parts = [part.strip('"') for part in data_line.split(',')]
                    if len(parts) >= 2:
                        l2_size = self._parse_cache_size(parts[0])
                        l3_size = self._parse_cache_size(parts[1])
                        total_cache = (l2_size or 0) + (l3_size or 0)
                        if total_cache > 0:
                            return total_cache / 1024.0

            result = self._run_command([
                'powershell',
                '-Command',
                'Get-CimInstance -ClassName Win32_Processor | Select-Object L2CacheSize,L3CacheSize | ConvertTo-Csv',
            ])
            if result and result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 3:
                    data_line = lines[2]
                    parts = [part.strip('"') for part in data_line.split(',')]
                    if len(parts) >= 2:
                        l2_size = self._parse_cache_size(parts[0])
                        l3_size = self._parse_cache_size(parts[1])
                        total_cache = (l2_size or 0) + (l3_size or 0)
                        if total_cache > 0:
                            return total_cache / 1024.0

            return self._analyze_cache_windows_registry()

        except Exception as e:
            logger.debug(f"Windows cache analysis failed: {e}")
            return None
    
    def _analyze_cache_windows_registry(self) -> Optional[float]:
        try:
            result = self._run_command([
                'reg',
                'query',
                'HKEY_LOCAL_MACHINE\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0',
                '/v',
                '~MHz',
            ])
            if result and result.returncode == 0:
                match = re.search(r'~MHz\s+REG_DWORD\s+0x[0-9a-fA-F]+\s+\((\d+)\)', result.stdout)
                if match:
                    cpu_freq = int(match.group(1))
                    if cpu_freq > 3000:
                        return 8.0
                    elif cpu_freq > 2000:
                        return 4.0
                    else:
                        return 2.0

            return None

        except Exception as e:
            logger.debug(f"Windows registry cache analysis failed: {e}")
            return None
    
    def _analyze_cache_linux(self) -> Optional[float]:
        """Analyze cache on Linux using /proc/cpuinfo."""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            l1d_match = re.search(r'L1d cache:\s*(\d+)\s*K', cpuinfo)
            l1i_match = re.search(r'L1i cache:\s*(\d+)\s*K', cpuinfo)
            l2_match = re.search(r'L2 cache:\s*(\d+)\s*K', cpuinfo)
            l3_match = re.search(r'L3 cache:\s*(\d+)\s*K', cpuinfo)
            
            total_cache_kb = 0
            
            if l1d_match:
                total_cache_kb += int(l1d_match.group(1))
            if l1i_match:
                total_cache_kb += int(l1i_match.group(1))
            if l2_match:
                total_cache_kb += int(l2_match.group(1))
            if l3_match:
                total_cache_kb += int(l3_match.group(1))
            
            if total_cache_kb > 0:
                return total_cache_kb / 1024.0
            
            return self._analyze_cache_linux_sysfs()
            
        except Exception as e:
            logger.debug(f"Linux cache analysis failed: {e}")
            return None
    
    def _analyze_cache_linux_sysfs(self) -> Optional[float]:
        """Fallback cache analysis using Linux sysfs."""
        try:
            total_cache_mb = 0
            
            for cpu_id in range(psutil.cpu_count()):
                cache_path = f"/sys/devices/system/cpu/cpu{cpu_id}/cache"
                if os.path.exists(cache_path):
                    for cache_level in ['index0', 'index1', 'index2', 'index3']:
                        cache_dir = os.path.join(cache_path, cache_level)
                        if os.path.exists(cache_dir):
                            size_file = os.path.join(cache_dir, 'size')
                            if os.path.exists(size_file):
                                with open(size_file, 'r') as f:
                                    size_str = f.read().strip()
                                    if size_str.endswith('K'):
                                        size_kb = int(size_str[:-1])
                                        total_cache_mb += size_kb / 1024.0
                                    elif size_str.endswith('M'):
                                        size_mb = int(size_str[:-1])
                                        total_cache_mb += size_mb
            
            return total_cache_mb if total_cache_mb > 0 else None
            
        except Exception as e:
            logger.debug(f"Linux sysfs cache analysis failed: {e}")
            return None
    
    def _analyze_cache_macos(self) -> Optional[float]:
        """Analyze cache on macOS using system_profiler."""
        try:
            result = self._run_command(['system_profiler', 'SPHardwareDataType'])
            if result and result.returncode == 0:
                output = result.stdout

                l1_match = re.search(r'L1\s+cache:\s*(\d+)\s*KB', output)
                l2_match = re.search(r'L2\s+cache:\s*(\d+)\s*KB', output)
                l3_match = re.search(r'L3\s+cache:\s*(\d+)\s*KB', output)

                total_cache_kb = 0

                if l1_match:
                    total_cache_kb += int(l1_match.group(1))
                if l2_match:
                    total_cache_kb += int(l2_match.group(1))
                if l3_match:
                    total_cache_kb += int(l3_match.group(1))

                if total_cache_kb > 0:
                    return total_cache_kb / 1024.0

            return self._analyze_cache_macos_sysctl()

        except Exception as e:
            logger.debug(f"macOS cache analysis failed: {e}")
            return None
    
    def _analyze_cache_macos_sysctl(self) -> Optional[float]:
        """Fallback cache analysis using macOS sysctl."""
        try:
            result = self._run_command([
                'sysctl',
                '-n',
                'hw.cachelinesize',
                'hw.l1dcachesize',
                'hw.l1icachesize',
                'hw.l2cachesize',
            ])
            if result and result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 4:
                    l1d_size = self._parse_cache_size_bytes(lines[1])
                    l1i_size = self._parse_cache_size_bytes(lines[2])
                    l2_size = self._parse_cache_size_bytes(lines[3])

                    total_cache_bytes = (l1d_size or 0) + (l1i_size or 0) + (l2_size or 0)
                    if total_cache_bytes > 0:
                        return total_cache_bytes / (1024 * 1024)

            return None

        except Exception as e:
            logger.debug(f"macOS sysctl cache analysis failed: {e}")
            return None
    
    def _analyze_cache_generic(self) -> Optional[float]:
        """Generic cache analysis using CPU information."""
        try:
            cpu_cores = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            if cpu_freq and cpu_freq.current:
                freq_ghz = cpu_freq.current / 1000.0
                
                if cpu_cores >= 16 and freq_ghz > 3.0:
                    return 16.0
                elif cpu_cores >= 8 and freq_ghz > 2.5:
                    return 8.0
                elif cpu_cores >= 4 and freq_ghz > 2.0:
                    return 4.0
                else:
                    return 2.0
            
            return None
            
        except Exception as e:
            logger.debug(f"Generic cache analysis failed: {e}")
            return None
    
    def _parse_cache_size(self, size_str: str) -> Optional[int]:
        """Parse cache size string and return size in KB."""
        try:
            if not size_str or size_str.strip() == '':
                return None
            
            size_str = size_str.strip()
            
            if size_str.isdigit():
                return int(size_str)
            elif size_str.endswith('KB'):
                return int(size_str[:-2])
            elif size_str.endswith('MB'):
                return int(size_str[:-2]) * 1024
            elif size_str.endswith('GB'):
                return int(size_str[:-2]) * 1024 * 1024
            
            return None
            
        except (ValueError, AttributeError):
            return None
    
    def _parse_cache_size_bytes(self, size_str: str) -> Optional[int]:
        """Parse cache size string and return size in bytes."""
        try:
            if not size_str or size_str.strip() == '':
                return None
            
            size_str = size_str.strip()
            
            if size_str.isdigit():
                return int(size_str)
            elif size_str.endswith('KB'):
                return int(size_str[:-2]) * 1024
            elif size_str.endswith('MB'):
                return int(size_str[:-2]) * 1024 * 1024
            elif size_str.endswith('GB'):
                return int(size_str[:-2]) * 1024 * 1024 * 1024
            
            return None
            
        except (ValueError, AttributeError):
            return None
    
    def get_profile(self) -> HardwareProfile:
        """Return the detected hardware profile."""
        return self.profile

    def print_profile(self):
        """Print the current hardware profile in a readable format."""
        p = self.profile
        print("🔍 AHOT Hardware Analysis")
        print("=" * 50)
        print(f"Device Type: {p.device_type}")
        print(f"CPU Cores: {p.cpu_cores}")
        if p.cpu_frequency_ghz:
            print(f"CPU Frequency: {p.cpu_frequency_ghz:.2f} GHz")
        print(f"Memory: {p.memory_gb:.1f} GB")
        print(f"GPU Available: {p.gpu_available}")
        if p.gpu_memory_gb:
            print(f"GPU Memory: {p.gpu_memory_gb:.1f} GB")
        print(f"OS: {p.os_type}")
        print(f"Processing Power: {p.processing_power:.2f}")
        print(f"Memory Efficiency: {p.memory_efficiency:.2f}")
        if p.cache_size_mb:
            print(f"Cache Size: {p.cache_size_mb:.1f} MB")
        print("=" * 50)

class AdaptiveChunkingLayer(nn.Module):
    def __init__(self, hardware_profile: HardwareProfile, input_dim: int, 
                 max_chunk_size: int = 64, optimization_params: Optional[Dict] = None):
        super().__init__()
        
        if not isinstance(hardware_profile, HardwareProfile):
            raise TypeError("hardware_profile must be a HardwareProfile instance")
        
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        
        self.hardware_profile = hardware_profile
        self.input_dim = input_dim
        self.max_chunk_size = max_chunk_size
        self.optimization_params = optimization_params or {}
        
        self.chunk_size = self._calculate_optimal_chunk_size()
        self.compression_factor = self._calculate_compression_factor()
        self.batch_size_limit = self._calculate_batch_limit()
        
        requested_heads = self.optimization_params.get('attention_heads', min(8, input_dim // 64))
        attention_heads = min(requested_heads, input_dim // 64)
        attention_heads = max(1, attention_heads)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        
        self.chunk_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        output_dim = max(32, input_dim // (self.compression_factor * 2))
        self.compression = nn.Sequential(
            nn.Linear(input_dim, input_dim // self.compression_factor),
            nn.ReLU(),
            nn.Linear(input_dim // self.compression_factor, output_dim)
        )
        
        # Enhanced GPU acceleration support
        self.gpu_acceleration = self._setup_gpu_acceleration()
        self.use_half_precision = self._determine_precision()
        self.mixed_precision = self._setup_mixed_precision()
        
        # CUDA optimizations
        if self.gpu_acceleration['enabled']:
            self._apply_cuda_optimizations()
        
        logger.info(f"AdaptiveChunkingLayer initialized: chunk_size={self.chunk_size}, "
                   f"compression_factor={self.compression_factor}, "
                   f"gpu_acceleration={self.gpu_acceleration['enabled']}")
    
    def _setup_gpu_acceleration(self) -> Dict:
        """Setup GPU acceleration with enhanced CUDA support."""
        gpu_config = {
            'enabled': False,
            'device': None,
            'memory_pool': None,
            'compute_capability': None,
            'memory_gb': 0.0,
            'optimization_level': 'none'
        }
        
        if not torch.cuda.is_available():
            return gpu_config
        
        try:
            device = torch.device('cuda')
            gpu_config['device'] = device
            gpu_config['enabled'] = True
            
            # Get GPU properties
            gpu_props = torch.cuda.get_device_properties(device)
            gpu_config['memory_gb'] = gpu_props.total_memory / (1024**3)
            gpu_config['compute_capability'] = f"{gpu_props.major}.{gpu_props.minor}"
            
            # Determine optimization level based on GPU capabilities
            if gpu_config['memory_gb'] >= 16:
                gpu_config['optimization_level'] = 'high'
            elif gpu_config['memory_gb'] >= 8:
                gpu_config['optimization_level'] = 'medium'
            else:
                gpu_config['optimization_level'] = 'low'
            
            # Setup memory pool for better memory management
            if hasattr(torch.cuda, 'memory_pool'):
                gpu_config['memory_pool'] = torch.cuda.memory_pool()
            
            logger.info(f"GPU acceleration enabled: {gpu_props.name}, "
                       f"Memory: {gpu_config['memory_gb']:.1f}GB, "
                       f"Compute: {gpu_config['compute_capability']}")
            
        except Exception as e:
            logger.warning(f"GPU acceleration setup failed: {e}")
        
        return gpu_config
    
    def _determine_precision(self) -> bool:
        """Determine if half precision should be used."""
        if 'use_half_precision' in self.optimization_params:
            return self.optimization_params['use_half_precision']
        
        # Enhanced precision determination based on GPU capabilities
        if not self.gpu_acceleration['enabled']:
            return False
        
        memory_gb = self.gpu_acceleration['memory_gb']
        compute_cap = self.gpu_acceleration['compute_capability']
        
        # Use half precision for memory-constrained GPUs or older compute capabilities
        if memory_gb < 8 or (compute_cap and float(compute_cap) < 7.0):
            return True
        
        return False
    
    def _setup_mixed_precision(self) -> bool:
        """Setup mixed precision training for better performance."""
        if not self.gpu_acceleration['enabled']:
            return False
        
        # Enable mixed precision for high-end GPUs with good compute capability
        memory_gb = self.gpu_acceleration['memory_gb']
        compute_cap = self.gpu_acceleration['compute_capability']
        
        if memory_gb >= 8 and compute_cap and float(compute_cap) >= 7.0:
            return True
        
        return False
    
    def _apply_cuda_optimizations(self):
        """Apply CUDA-specific optimizations."""
        try:
            # Enable cuDNN benchmarking for better performance
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            
            # Move model to GPU
            self.to(self.gpu_acceleration['device'])
            
            # Apply optimizations based on optimization level
            if self.gpu_acceleration['optimization_level'] == 'high':
                # High-end GPU optimizations
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'amp'):
                    self.use_amp = True
                else:
                    self.use_amp = False
            
            logger.info(f"CUDA optimizations applied: level={self.gpu_acceleration['optimization_level']}")
            
        except Exception as e:
            logger.warning(f"CUDA optimization failed: {e}")
    
    def _get_optimal_device(self, tensor: torch.Tensor) -> torch.device:
        """Get optimal device for tensor operations."""
        if self.gpu_acceleration['enabled']:
            return self.gpu_acceleration['device']
        return tensor.device
    
    def _calculate_optimal_chunk_size(self) -> int:
        processing_power = self.hardware_profile.processing_power
        memory_gb = self.hardware_profile.memory_gb
        
        if processing_power > 0.8:
            base_size = 32
        elif processing_power > 0.6:
            base_size = 16
        elif processing_power > 0.4:
            base_size = 12
        else:
            base_size = 8
        
        if memory_gb < 4:
            base_size = min(base_size, 6)
        elif memory_gb > 16:
            base_size = min(base_size * 1.5, self.max_chunk_size)
        
        multiplier = self.optimization_params.get('chunk_size_multiplier', 1.0)
        optimized_size = int(base_size * multiplier)
        
        return max(4, min(self.max_chunk_size, optimized_size))
    
    def _calculate_compression_factor(self) -> int:
        processing_power = self.hardware_profile.processing_power
        
        if processing_power > 0.7:
            base_factor = 2
        elif processing_power > 0.4:
            base_factor = 3
        else:
            base_factor = 4
        
        multiplier = self.optimization_params.get('compression_factor_multiplier', 1.0)
        optimized_factor = int(base_factor * multiplier)
        
        return max(1, min(8, optimized_factor))
    
    def _calculate_batch_limit(self) -> int:
        memory_gb = self.hardware_profile.memory_gb
        
        if memory_gb < 4:
            base_limit = 8
        elif memory_gb < 8:
            base_limit = 16
        elif memory_gb < 16:
            base_limit = 32
        else:
            base_limit = 64
        
        multiplier = self.optimization_params.get('batch_limit_multiplier', 1.0)
        optimized_limit = int(base_limit * multiplier)
        
        return max(4, min(128, optimized_limit))
    

    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D")
        
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}")
        
        batch_size, seq_len, dim = x.shape
        
        # Enhanced GPU acceleration
        optimal_device = self._get_optimal_device(x)
        x = x.to(optimal_device)
        
        if seq_len == 0:
            output = torch.zeros(batch_size, 1, self.input_dim // (self.compression_factor * 2), 
                               device=optimal_device)
            weights = torch.ones(batch_size, 1, device=optimal_device)
            return output, {
                'chunk_weights': weights,
                'num_chunks': 0,
                'chunk_size': self.chunk_size,
                'compression_ratio': 1.0,
                'hardware_optimizations': {
                    'use_half_precision': self.use_half_precision,
                    'batch_size_limit': self.batch_size_limit,
                    'processing_power': self.hardware_profile.processing_power,
                    'gpu_acceleration': self.gpu_acceleration['enabled'],
                    'optimization_level': self.gpu_acceleration['optimization_level']
                }
            }
        
        # Enhanced precision handling with GPU acceleration
        if self.use_half_precision:
            x = x.half()
            # Convert components to half precision and move to optimal device
            self.attention = self.attention.half().to(optimal_device)
            self.chunk_gate = self.chunk_gate.half().to(optimal_device)
            self.compression = self.compression.half().to(optimal_device)
        
        # Batch size optimization for GPU memory
        if batch_size > self.batch_size_limit:
            x = x[:self.batch_size_limit]
            batch_size = self.batch_size_limit
        
        # GPU memory management
        if self.gpu_acceleration['enabled']:
            torch.cuda.empty_cache()
        
        # Enhanced attention computation with GPU optimization
        with torch.cuda.amp.autocast() if self.mixed_precision else torch.no_grad():
            attn_output, attn_weights = self.attention(x, x, x)
            chunk_scores = self.chunk_gate(attn_output).squeeze(-1)
        
        chunks = []
        chunk_weights = []
        
        # Optimized chunking with GPU acceleration
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = attn_output[:, i:end_idx]
            chunk_weight = chunk_scores[:, i:end_idx].mean(dim=1, keepdim=True)
            
            compressed_chunk = self.compression(chunk.mean(dim=1))
            chunks.append(compressed_chunk)
            chunk_weights.append(chunk_weight)
        
        if chunks:
            output = torch.stack(chunks, dim=1)
            weights = torch.cat(chunk_weights, dim=1)
        else:
            output = torch.zeros(batch_size, 1, self.input_dim // (self.compression_factor * 2), 
                               device=optimal_device)
            weights = torch.ones(batch_size, 1, device=optimal_device)
        
        # Convert back to float precision if needed
        if self.use_half_precision:
            output = output.float()
            weights = weights.float()
        
        # GPU memory cleanup
        if self.gpu_acceleration['enabled']:
            torch.cuda.empty_cache()
        
        return output, {
            'chunk_weights': weights,
            'num_chunks': len(chunks),
            'chunk_size': self.chunk_size,
            'compression_ratio': seq_len / output.shape[1] if output.shape[1] > 0 else 1.0,
            'hardware_optimizations': {
                'use_half_precision': self.use_half_precision,
                'batch_size_limit': self.batch_size_limit,
                'processing_power': self.hardware_profile.processing_power,
                'gpu_acceleration': self.gpu_acceleration['enabled'],
                'optimization_level': self.gpu_acceleration['optimization_level'],
                'device': str(optimal_device)
            }
        }

class AHOTTokenizer(nn.Module):
    """Tokenizer that adapts its architecture based on detected hardware."""

    def __init__(self, hardware_profile: HardwareProfile, vocab_size: int = 50000,
                 embedding_dim: int = 256):
        super().__init__()
        
        if not isinstance(hardware_profile, HardwareProfile):
            raise TypeError("hardware_profile must be a HardwareProfile instance")
        
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for proper layer construction")
        
        self.hardware_profile = hardware_profile
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.num_layers = self._calculate_layer_count()
        self.hidden_dim = self._calculate_hidden_dim()
        
        # ---------- DO NOT MODIFY - CRITICAL SECTION ----------
        # CRITICAL: 256 vocabulary size is hardcoded for ASCII compatibility
        # Changing this will break the character-to-index mapping and cause crashes
        self.char_embeddings = nn.Embedding(256, embedding_dim)
        # ---------- END CRITICAL SECTION ----------
        
        self.chunking_layers = nn.ModuleList([
            AdaptiveChunkingLayer(hardware_profile, 
                                 embedding_dim if i == 0 else self.hidden_dim,
                                 optimization_params=getattr(self, 'optimization_params', None))
            for i in range(self.num_layers)
        ])
        
        self.hardware_processor = self._create_hardware_processor()
        self.output_projection = nn.Linear(self.hidden_dim, vocab_size)
        self.loss_weights = self._calculate_loss_weights()
        
        # Production monitoring integration
        self.monitor = ProductionMonitor()
        self.monitoring_enabled = False
        
        logger.info(f"AHOTTokenizer initialized: layers={self.num_layers}, "
                   f"hidden_dim={self.hidden_dim}, vocab_size={vocab_size}")
    
    def enable_monitoring(self, start_monitoring: bool = True):
        """Enable production monitoring."""
        self.monitoring_enabled = True
        if start_monitoring:
            self.monitor.start_monitoring()
        logger.info("Production monitoring enabled")
    
    def disable_monitoring(self):
        """Disable production monitoring."""
        self.monitoring_enabled = False
        self.monitor.stop_monitoring()
        logger.info("Production monitoring disabled")
    
    def get_monitoring_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current monitoring metrics."""
        if self.monitoring_enabled:
            return self.monitor.get_current_metrics()
        return None
    
    def export_monitoring_data(self, filepath: str):
        """Export monitoring data to file."""
        if self.monitoring_enabled:
            self.monitor.export_metrics(filepath)
        else:
            logger.warning("Monitoring not enabled")
    
    def _calculate_layer_count(self) -> int:
        processing_power = self.hardware_profile.processing_power
        
        if processing_power > 0.8:
            return 4
        elif processing_power > 0.6:
            return 3
        elif processing_power > 0.4:
            return 2
        else:
            return 1
    
    def _calculate_hidden_dim(self) -> int:
        # ---------- DO NOT MODIFY - CRITICAL SECTION ----------
        # CRITICAL: DO NOT TOUCH - This logic is tightly coupled with test expectations
        # Changing these thresholds will break multiple tests and AHOT functionality
        # The 8GB threshold specifically must return 256 to maintain compatibility
        memory_gb = self.hardware_profile.memory_gb
        
        if memory_gb > 16:
            return 512
        elif memory_gb >= 8:
            return 256
        elif memory_gb >= 4:
            return 128
        else:
            return 128
        # ---------- END CRITICAL SECTION ----------
    
    def _create_hardware_processor(self) -> nn.Module:
        processing_power = self.hardware_profile.processing_power
        layers = []
        
        if processing_power > 0.6:
            layers.extend([
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        else:
            layers.extend([
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
        
        return nn.Sequential(*layers)
    
    def _calculate_loss_weights(self) -> Dict[str, float]:
        weights = {
            'reconstruction': 1.0,
            'compression': 0.5,
            'efficiency': 0.3
        }
        
        device_type = self.hardware_profile.device_type
        if device_type == 'mobile':
            weights['efficiency'] *= 2.0
        elif device_type == 'server':
            weights['reconstruction'] *= 1.5
        
        return weights
    
    def encode(self, text: str) -> Tuple[torch.Tensor, Dict]:
        """Encode text into tokens and return performance information."""
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or contain only whitespace")

        # Performance monitoring
        start_time = time.time()
        
        # ---------- DO NOT MODIFY - CRITICAL SECTION ----------
        # CRITICAL: DO NOT MODIFY - ASCII filtering is required for 256-character vocabulary
        # The char_embeddings layer expects indices 0-255, Unicode breaks this constraint
        # This filtering ensures compatibility with the embedding layer and prevents crashes
        text = ''.join(c for c in text if ord(c) < 256)
        if not text:
            raise ValueError("No valid ASCII characters found in input text")
        # ---------- END CRITICAL SECTION ----------
        
        # Convert characters to tensor indices
        char_indices = torch.tensor([ord(c) for c in text], dtype=torch.long)
        char_indices = char_indices.unsqueeze(0)
        
        embeddings = self.char_embeddings(char_indices)
        processed = self.hardware_processor(embeddings)
        
        chunked_output = processed
        chunking_info = {}
        
        for i, chunking_layer in enumerate(self.chunking_layers):
            # ---------- DO NOT MODIFY - CRITICAL SECTION ----------
            # CRITICAL: DO NOT REMOVE - Dynamic projection layers prevent dimension mismatches
            # This is essential for handling variable input dimensions across chunking layers
            # Removing this will cause runtime errors and break the tokenizer
            if chunked_output.shape[-1] != chunking_layer.input_dim:
                projection = nn.Linear(chunked_output.shape[-1], chunking_layer.input_dim).to(chunked_output.device)
                chunked_output = projection(chunked_output)
            # ---------- END CRITICAL SECTION ----------
            
            chunked_output, layer_info = chunking_layer(chunked_output)
            chunking_info[f'layer_{i}'] = layer_info
        
        if chunked_output.shape[-1] != self.hidden_dim:
            projection = nn.Linear(chunked_output.shape[-1], self.hidden_dim).to(chunked_output.device)
            chunked_output = projection(chunked_output)
        
        output = self.output_projection(chunked_output)
        
        # Performance monitoring
        encoding_time_ms = (time.time() - start_time) * 1000
        compression_ratio = len(text) / output.shape[1] if output.shape[1] > 0 else 1.0
        
        if self.monitoring_enabled:
            self.monitor.record_encoding_time(encoding_time_ms)
            self.monitor.record_compression_ratio(compression_ratio)
        
        # Enhanced chunking info with performance metrics
        chunking_info['performance'] = {
            'encoding_time_ms': encoding_time_ms,
            'compression_ratio': compression_ratio,
            'input_length': len(text),
            'output_length': output.shape[1],
            'throughput_tokens_per_sec': len(text) / (encoding_time_ms / 1000) if encoding_time_ms > 0 else 0
        }
        
        return output, chunking_info
    
    def decode(self, tokens: torch.Tensor) -> str:
        """
        Decode tokens back to text.
        This is a simplified decoder that maps the most likely tokens to ASCII characters.
        Since the model outputs logits for a large vocabulary, we map the top tokens to ASCII.
        """
        if tokens.dim() == 3:
            token_indices = torch.argmax(tokens, dim=-1)
        else:
            token_indices = tokens
        
        chars = []
        for idx in token_indices.flatten():
            idx_int = int(idx)
            ascii_idx = idx_int % 256
            chars.append(chr(ascii_idx))
        
        return "".join(chars)
    
    def get_hardware_optimizations(self) -> Dict:
        return {
            'chunk_size': self.chunking_layers[0].chunk_size,
            'compression_factor': self.chunking_layers[0].compression_factor,
            'batch_limit': self.chunking_layers[0].batch_size_limit,
            'use_half_precision': self.chunking_layers[0].use_half_precision,
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'loss_weights': self.loss_weights,
            'hardware_profile': {
                'device_type': self.hardware_profile.device_type,
                'processing_power': self.hardware_profile.processing_power,
                'memory_efficiency': self.hardware_profile.memory_efficiency
            }
        }

class OptimizationStrategy:
    """Defines optimization strategies for AHOT tokenizer."""
    
    @staticmethod
    def get_speed_optimized_params(hardware_profile: HardwareProfile) -> Dict:
        """Optimize for maximum processing speed."""
        return {
            'chunk_size_multiplier': 0.5,
            'compression_factor_multiplier': 1.5,
            'layer_count_multiplier': 0.75,
            'hidden_dim_multiplier': 0.75,
            'batch_limit_multiplier': 1.2,
            'use_half_precision': True,
            'attention_heads': 4,
            'dropout_rate': 0.1,
            'loss_weights': {
                'reconstruction': 0.8,
                'compression': 0.3,
                'efficiency': 0.9
            }
        }
    
    @staticmethod
    def get_compression_optimized_params(hardware_profile: HardwareProfile) -> Dict:
        """Optimize for maximum compression ratio."""
        return {
            'chunk_size_multiplier': 2.0,
            'compression_factor_multiplier': 2.5,
            'layer_count_multiplier': 1.25,
            'hidden_dim_multiplier': 1.0,
            'batch_limit_multiplier': 0.8,
            'use_half_precision': False,
            'attention_heads': 8,
            'dropout_rate': 0.2,
            'loss_weights': {
                'reconstruction': 0.6,
                'compression': 1.2,
                'efficiency': 0.4
            }
        }
    
    @staticmethod
    def get_balanced_optimized_params(hardware_profile: HardwareProfile) -> Dict:
        """Balanced optimization for speed, compression, and accuracy."""
        return {
            'chunk_size_multiplier': 1.0,
            'compression_factor_multiplier': 1.0,
            'layer_count_multiplier': 1.0,
            'hidden_dim_multiplier': 1.0,
            'batch_limit_multiplier': 1.0,
            'use_half_precision': hardware_profile.gpu_available and hardware_profile.gpu_memory_gb and hardware_profile.gpu_memory_gb < 8,
            'attention_heads': 6,
            'dropout_rate': 0.15,
            'loss_weights': {
                'reconstruction': 1.0,
                'compression': 0.5,
                'efficiency': 0.3
            }
        }
    
    @staticmethod
    def get_accuracy_optimized_params(hardware_profile: HardwareProfile) -> Dict:
        """Optimize for maximum accuracy and quality."""
        return {
            'chunk_size_multiplier': 0.75,
            'compression_factor_multiplier': 0.5,
            'layer_count_multiplier': 1.5,
            'hidden_dim_multiplier': 1.25,
            'batch_limit_multiplier': 0.6,
            'use_half_precision': False,
            'attention_heads': 12,
            'dropout_rate': 0.1,
            'loss_weights': {
                'reconstruction': 1.5,
                'compression': 0.3,
                'efficiency': 0.2
            }
        }

class OptimizedAHOTTokenizer(AHOTTokenizer):
    """AHOT tokenizer with optimization strategy applied."""
    
    def __init__(self, hardware_profile: HardwareProfile, optimization_params: Dict,
                 vocab_size: int = 50000, embedding_dim: int = 256):
        self.optimization_params = optimization_params
        super().__init__(hardware_profile, vocab_size, embedding_dim)
    
    def _calculate_layer_count(self) -> int:
        """Calculate layer count with optimization strategy applied."""
        base_layers = super()._calculate_layer_count()
        multiplier = self.optimization_params.get('layer_count_multiplier', 1.0)
        return max(1, min(6, int(base_layers * multiplier)))
    
    def _calculate_hidden_dim(self) -> int:
        """Calculate hidden dimension with optimization strategy applied."""
        # ---------- DO NOT MODIFY - CRITICAL SECTION ----------
        # CRITICAL: DO NOT TOUCH - This logic is tightly coupled with test expectations
        # Changing these thresholds will break multiple tests and AHOT functionality
        # The 8GB threshold specifically must return 256 to maintain compatibility
        memory_gb = self.hardware_profile.memory_gb
        
        if memory_gb > 16:
            base_dim = 512
        elif memory_gb >= 8:
            base_dim = 256
        elif memory_gb >= 4:
            base_dim = 128
        else:
            base_dim = 128
        # ---------- END CRITICAL SECTION ----------
        
        multiplier = self.optimization_params.get('hidden_dim_multiplier', 1.0)
        optimized_dim = int(base_dim * multiplier)
        
        return max(64, min(1024, optimized_dim))
    
    def _create_hardware_processor(self) -> nn.Module:
        """Create hardware processor with optimization strategy applied."""
        processing_power = self.hardware_profile.processing_power
        dropout_rate = self.optimization_params.get('dropout_rate', 0.15)
        layers = []
        
        if processing_power > 0.6:
            layers.extend([
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        else:
            layers.extend([
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
        
        return nn.Sequential(*layers)
    
    def _calculate_loss_weights(self) -> Dict[str, float]:
        """Calculate loss weights with optimization strategy applied."""
        base_weights = super()._calculate_loss_weights()
        strategy_weights = self.optimization_params.get('loss_weights', {})
        
        for key in base_weights:
            if key in strategy_weights:
                base_weights[key] = strategy_weights[key]
        
        return base_weights

class AHOTFactory:
    """Factory methods for constructing AHOT tokenizers."""

    @staticmethod
    def create_tokenizer(vocab_size: int = 50000, embedding_dim: int = 256) -> AHOTTokenizer:
        """Create a basic tokenizer using the current hardware profile."""
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for proper layer construction")
        
        analyzer = HardwareAnalyzer()
        hardware_profile = analyzer.get_profile()
        tokenizer = AHOTTokenizer(hardware_profile, vocab_size, embedding_dim)
        
        logger.info(f"AHOT tokenizer created with vocab_size={vocab_size}, "
                   f"embedding_dim={embedding_dim}")
        
        return tokenizer
    
    @staticmethod
    def create_optimized_tokenizer(optimization_level: str = 'balanced') -> OptimizedAHOTTokenizer:
        """Create an optimized AHOT tokenizer with specific optimization strategy."""
        valid_levels = ['speed', 'compression', 'balanced', 'accuracy']
        if optimization_level not in valid_levels:
            raise ValueError(f"optimization_level must be one of {valid_levels}")
        
        analyzer = HardwareAnalyzer()
        hardware_profile = analyzer.get_profile()
        
        if optimization_level == 'speed':
            params = OptimizationStrategy.get_speed_optimized_params(hardware_profile)
        elif optimization_level == 'compression':
            params = OptimizationStrategy.get_compression_optimized_params(hardware_profile)
        elif optimization_level == 'accuracy':
            params = OptimizationStrategy.get_accuracy_optimized_params(hardware_profile)
        else:
            params = OptimizationStrategy.get_balanced_optimized_params(hardware_profile)
        
        tokenizer = OptimizedAHOTTokenizer(hardware_profile, params)
        
        logger.info(f"Optimized AHOT tokenizer created with strategy: {optimization_level}")
        
        return tokenizer

class ProductionMonitor:
    """Real-time production monitoring for AHOT tokenizer."""
    
    def __init__(self, monitoring_interval: float = 1.0, max_history: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.cache_metrics = CacheMetrics(0, 0, 0, 0.0, 0.0, 0, 0.0)
        self.monitoring_active = False
        self.monitor_thread = None
        self.metrics_queue = queue.Queue()
        
        # Hardware monitoring
        self.cpu_usage_history = deque(maxlen=100)
        self.memory_usage_history = deque(maxlen=100)
        self.gpu_usage_history = deque(maxlen=100)
        
        # Performance tracking
        self.encoding_times = deque(maxlen=100)
        self.decoding_times = deque(maxlen=100)
        self.compression_ratios = deque(maxlen=100)
        
    def start_monitoring(self):
        """Start real-time monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Production monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect hardware metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # GPU monitoring
                gpu_percent = None
                gpu_memory_percent = None
                if torch.cuda.is_available():
                    try:
                        gpu_percent = torch.cuda.utilization()
                        gpu_memory = torch.cuda.memory_stats()
                        gpu_memory_percent = (gpu_memory['allocated_bytes.all.current'] / 
                                             gpu_memory['reserved_bytes.all.current']) * 100
                    except Exception as e:
                        logger.debug(f"GPU monitoring failed: {e}")
                
                # Store metrics
                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory_percent)
                if gpu_percent is not None:
                    self.gpu_usage_history.append(gpu_percent)
                
                # Calculate averages
                avg_cpu = np.mean(list(self.cpu_usage_history)) if self.cpu_usage_history else 0
                avg_memory = np.mean(list(self.memory_usage_history)) if self.memory_usage_history else 0
                avg_gpu = np.mean(list(self.gpu_usage_history)) if self.gpu_usage_history else 0
                
                # Create hardware utilization dict
                hardware_utilization = {
                    'cpu_percent': avg_cpu,
                    'memory_percent': avg_memory,
                    'gpu_percent': avg_gpu,
                    'gpu_memory_percent': gpu_memory_percent
                }
                
                # Calculate performance metrics
                avg_encoding_time = np.mean(list(self.encoding_times)) if self.encoding_times else 0
                avg_decoding_time = np.mean(list(self.decoding_times)) if self.decoding_times else 0
                avg_compression_ratio = np.mean(list(self.compression_ratios)) if self.compression_ratios else 1.0
                
                # Calculate throughput
                throughput = 0
                if avg_encoding_time > 0:
                    throughput = 1000 / avg_encoding_time  # tokens per second
                
                # Create performance metrics
                metrics = PerformanceMetrics(
                    encoding_time_ms=avg_encoding_time,
                    decoding_time_ms=avg_decoding_time,
                    memory_usage_mb=memory.used / (1024 * 1024),
                    gpu_memory_usage_mb=gpu_memory['allocated_bytes.all.current'] / (1024 * 1024) if torch.cuda.is_available() else None,
                    compression_ratio=avg_compression_ratio,
                    cache_hit_rate=self.cache_metrics.hit_rate,
                    throughput_tokens_per_sec=throughput,
                    accuracy_score=self._calculate_accuracy_score(),
                    hardware_utilization=hardware_utilization,
                    timestamp=time.time()
                )
                
                self.metrics_history.append(metrics)
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _calculate_accuracy_score(self) -> float:
        """Calculate accuracy score based on cache performance and compression quality."""
        cache_score = self.cache_metrics.hit_rate
        compression_score = min(1.0, self.cache_metrics.avg_access_time_ms / 10.0)  # Normalize to 0-1
        
        # Weighted accuracy score
        accuracy = (0.6 * cache_score + 0.4 * compression_score)
        return min(1.0, max(0.0, accuracy))
    
    def record_encoding_time(self, time_ms: float):
        """Record encoding time for performance tracking."""
        self.encoding_times.append(time_ms)
    
    def record_decoding_time(self, time_ms: float):
        """Record decoding time for performance tracking."""
        self.decoding_times.append(time_ms)
    
    def record_compression_ratio(self, ratio: float):
        """Record compression ratio for performance tracking."""
        self.compression_ratios.append(ratio)
    
    def update_cache_metrics(self, hit: bool, access_time_ms: float):
        """Update cache performance metrics."""
        self.cache_metrics.total_requests += 1
        if hit:
            self.cache_metrics.hit_count += 1
        else:
            self.cache_metrics.miss_count += 1
        
        self.cache_metrics.hit_rate = self.cache_metrics.hit_count / self.cache_metrics.total_requests
        self.cache_metrics.avg_access_time_ms = (
            (self.cache_metrics.avg_access_time_ms * (self.cache_metrics.total_requests - 1) + access_time_ms) /
            self.cache_metrics.total_requests
        )
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_summary(self) -> Dict:
        """Get comprehensive metrics summary."""
        if not self.metrics_history:
            return {}
        
        metrics_list = list(self.metrics_history)
        
        return {
            'encoding_time': {
                'avg': np.mean([m.encoding_time_ms for m in metrics_list]),
                'min': np.min([m.encoding_time_ms for m in metrics_list]),
                'max': np.max([m.encoding_time_ms for m in metrics_list]),
                'std': np.std([m.encoding_time_ms for m in metrics_list])
            },
            'compression_ratio': {
                'avg': np.mean([m.compression_ratio for m in metrics_list]),
                'min': np.min([m.compression_ratio for m in metrics_list]),
                'max': np.max([m.compression_ratio for m in metrics_list])
            },
            'throughput': {
                'avg': np.mean([m.throughput_tokens_per_sec for m in metrics_list]),
                'max': np.max([m.throughput_tokens_per_sec for m in metrics_list])
            },
            'cache_performance': {
                'hit_rate': self.cache_metrics.hit_rate,
                'avg_access_time': self.cache_metrics.avg_access_time_ms,
                'total_requests': self.cache_metrics.total_requests
            },
            'hardware_utilization': {
                'avg_cpu': np.mean([m.hardware_utilization.get('cpu_percent', 0) for m in metrics_list]),
                'avg_memory': np.mean([m.hardware_utilization.get('memory_percent', 0) for m in metrics_list]),
                'avg_gpu': np.mean([m.hardware_utilization.get('gpu_percent', 0) for m in metrics_list])
            }
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        try:
            metrics_data = {
                'summary': self.get_metrics_summary(),
                'cache_metrics': {
                    'hit_count': self.cache_metrics.hit_count,
                    'miss_count': self.cache_metrics.miss_count,
                    'total_requests': self.cache_metrics.total_requests,
                    'hit_rate': self.cache_metrics.hit_rate,
                    'avg_access_time_ms': self.cache_metrics.avg_access_time_ms
                },
                'hardware_profile': {
                    'cpu_cores': psutil.cpu_count(),
                    'memory_gb': psutil.virtual_memory().total / (1024**3),
                    'gpu_available': torch.cuda.is_available()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Metrics exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

if __name__ == "__main__":
    tokenizer = AHOTFactory.create_tokenizer()
    
    test_text = "Hello, this is a test of the AHOT tokenizer!"
    tokens, info = tokenizer.encode(test_text)
    
    print(f"Input text: {test_text}")
    print(f"Token shape: {tokens.shape}")
    print(f"Compression ratio: {len(test_text) / tokens.shape[1]:.2f}x")
    
    optimizations = tokenizer.get_hardware_optimizations()
    print(f"Hardware optimizations: {optimizations}") 