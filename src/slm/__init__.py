"""
Security-focused Small Language Model (SLM) components.

This module provides a custom transformer-based language model 
optimized for threat intelligence tasks including:
- Threat classification (malware, phishing, APT, etc.)
- Severity scoring (Critical/High/Medium/Low)
- MITRE ATT&CK mapping
- IOC extraction with BIO tagging
"""

from .tokenizer import SecurityTokenizer
from .model import (
    ThreatIntelSLM,
    ThreatIntelConfig,
    ThreatIntelInference,
    get_model_config,
)

__all__ = [
    "SecurityTokenizer",
    "ThreatIntelSLM",
    "ThreatIntelConfig",
    "ThreatIntelInference",
    "get_model_config",
]
