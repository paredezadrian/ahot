import sys, pathlib
import pytest

# Ensure package import
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from ahot import HardwareAnalyzer
from ahot.vocabulary import MemoryAwareVocabSizer


def test_classify_device_invalid_values():
    analyzer = HardwareAnalyzer()
    with pytest.raises(ValueError):
        analyzer._classify_device(-1, 4, False)
    with pytest.raises(ValueError):
        analyzer._classify_device(1, 0, False)


def test_processing_power_validation():
    analyzer = HardwareAnalyzer()
    with pytest.raises(ValueError):
        analyzer._calculate_processing_power(0, 1, False, None)
    with pytest.raises(ValueError):
        analyzer._calculate_processing_power(1, -1, False, None)
    with pytest.raises(ValueError):
        analyzer._calculate_processing_power(1, 1, False, 0)


def test_memory_efficiency_validation():
    analyzer = HardwareAnalyzer()
    with pytest.raises(ValueError):
        analyzer._calculate_memory_efficiency(-1, 'Linux')
    with pytest.raises(ValueError):
        analyzer._calculate_memory_efficiency(1, '')


def test_memory_aware_vocab_sizer_validation():
    sizer = MemoryAwareVocabSizer()
    with pytest.raises(ValueError):
        sizer.calculate_optimal_vocab_size(0, 128, False)
    with pytest.raises(ValueError):
        sizer.calculate_optimal_vocab_size(1, 0, False)


def test_memory_aware_vocab_sizer_range():
    sizer = MemoryAwareVocabSizer()
    size = sizer.calculate_optimal_vocab_size(8, 128, False)
    assert 16000 <= size <= 32000
