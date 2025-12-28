import pytest
import pandas as pd
import sys
import os

# Add src to path so we can import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import ip_to_int

def test_ip_to_int_standard_string():
    """Test standard x.x.x.x string format"""
    ip_series = pd.Series(["192.168.0.1"])
    result = ip_to_int(ip_series)
    # 192*(256^3) + 168*(256^2) + 0 + 1 = 3232235521
    assert result[0] == 3232235521

def test_ip_to_int_float_input():
    """Test if float inputs (common in fraud datasets) are cast to int"""
    ip_series = pd.Series([3232235521.0])
    result = ip_to_int(ip_series)
    assert result[0] == 3232235521

def test_ip_to_int_invalid():
    """Test invalid input returns 0"""
    ip_series = pd.Series(["invalid_ip"])
    result = ip_to_int(ip_series)
    assert result[0] == 0