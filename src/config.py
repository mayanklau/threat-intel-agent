"""
Configuration management for Threat Intelligence Agent.

Uses pydantic-settings for type-safe configuration with support for:
- Environment variables
- .env files
- YAML configuration files
- Secrets management
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SLMConfig(BaseSettings):
    """Configuration for the Small Language Model."""
    
    model_config = SettingsConfigDict(env_prefix="SLM_")
    
    model_size: Literal["small", "medium", "large"] = Field(
        default="medium",
        description="Model size: small (~30M), medium (~110M), large (~340M)"
    )
    vocab_size: int = Field(default=50000, ge=1000, le=200000)
    max_seq_length: int = Field(default=512, ge=64, le=4096)
    hidden_size: int = Field(default=768, ge=128)
    num_attention_heads: int = Field(default=12, ge=1)
    num_hidden_layers: int = Field(default=12, ge=1)
    intermediate_size: int = Field(default=3072, ge=256)
    dropout_prob: float = Field(default=0.1, ge=0.0, le=0.5)
    use_flash_attention: bool = Field(default=False)
    gradient_checkpointing: bool = Field(default=False)
    model_path: Path = Field(default=Path("data/models/threat-intel-slm"))
    device: str = Field(default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")


class MemoryConfig(BaseSettings):
    """Configuration for the multi-tier memory system."""
    
    model_config = SettingsConfigDict(env_prefix="MEMORY_")
    
    # Working Memory (Redis)
    redis_url: str = Field(default="redis://localhost:6379/0")
    redis_password: SecretStr | None = Field(default=None)
    working_memory_ttl: int = Field(default=3600, description="TTL in seconds for working memory")
    
    # Vector Store (ChromaDB)
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8000)
    chroma_persist_directory: Path = Field(default=Path("data/embeddings"))
    
    # Bead Memory
    bead_temporal_window: int = Field(
        default=86400 * 7,  # 7 days
        description="Time window in seconds for temporal correlation"
    )
    bead_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for bead correlation"
    )
    max_string_length: int = Field(default=100, description="Maximum beads per string")
    
    # Episodic Memory
    episodic_collection: str = Field(default="investigations")
    episodic_max_results: int = Field(default=10)
    
    # Semantic Memory  
    semantic_collection: str = Field(default="threat_knowledge")
    semantic_max_results: int = Field(default=20)


class ThreatFeedConfig(BaseSettings):
    """Configuration for threat intelligence feed providers."""
    
    model_config = SettingsConfigDict(env_prefix="THREATFEED_")
    
    # AlienVault OTX
    otx_api_key: SecretStr | None = Field(default=None)
    otx_base_url: str = Field(default="https://otx.alienvault.com")
    
    # VirusTotal
    virustotal_api_key: SecretStr | None = Field(default=None)
    virustotal_base_url: str = Field(default="https://www.virustotal.com/api/v3")
    
    # MISP
    misp_url: str | None = Field(default=None)
    misp_api_key: SecretStr | None = Field(default=None)
    misp_verify_ssl: bool = Field(default=True)
    
    # AbuseIPDB
    abuseipdb_api_key: SecretStr | None = Field(default=None)
    abuseipdb_base_url: str = Field(default="https://api.abuseipdb.com/api/v2")
    
    # STIX/TAXII
    taxii_url: str | None = Field(default=None)
    taxii_username: str | None = Field(default=None)
    taxii_password: SecretStr | None = Field(default=None)
    
    # Caching
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=10000)
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Requests per minute")
    rate_limit_period: int = Field(default=60, description="Rate limit period in seconds")
    
    # Timeouts
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3)


class APIConfig(BaseSettings):
    """Configuration for the REST API."""
    
    model_config = SettingsConfigDict(env_prefix="API_")
    
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    workers: int = Field(default=4, ge=1, le=32)
    reload: bool = Field(default=False)
    
    # Security
    cors_origins: list[str] = Field(default=["*"])
    api_key: SecretStr | None = Field(default=None, description="API key for authentication")
    rate_limit: int = Field(default=1000, description="Requests per minute per client")
    
    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: str = Field(default="json")
    
    # Metrics
    metrics_enabled: bool = Field(default=True)
    metrics_path: str = Field(default="/metrics")


class AgentConfig(BaseSettings):
    """Configuration for the threat intelligence agent."""
    
    model_config = SettingsConfigDict(env_prefix="AGENT_")
    
    # Processing
    max_concurrent_tasks: int = Field(default=10, ge=1)
    task_timeout: int = Field(default=300, description="Task timeout in seconds")
    batch_size: int = Field(default=100, ge=1, le=1000)
    
    # Enrichment
    enrichment_providers: list[str] = Field(
        default=["otx", "virustotal", "abuseipdb"],
        description="Active threat feed providers"
    )
    min_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Alert thresholds
    critical_severity_threshold: float = Field(default=0.9)
    high_severity_threshold: float = Field(default=0.7)
    medium_severity_threshold: float = Field(default=0.4)
    
    # Investigation
    auto_correlate: bool = Field(default=True)
    max_investigation_depth: int = Field(default=5)
    
    @field_validator("enrichment_providers")
    @classmethod
    def validate_providers(cls, v: list[str]) -> list[str]:
        valid_providers = {"otx", "virustotal", "misp", "abuseipdb", "taxii"}
        for provider in v:
            if provider not in valid_providers:
                raise ValueError(f"Invalid provider: {provider}. Valid: {valid_providers}")
        return v


class Settings(BaseSettings):
    """Main application settings aggregating all configurations."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    # Environment
    environment: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    
    # Sub-configurations
    slm: SLMConfig = Field(default_factory=SLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    threat_feeds: ThreatFeedConfig = Field(default_factory=ThreatFeedConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the application settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings (useful for testing)."""
    global _settings
    _settings = None
