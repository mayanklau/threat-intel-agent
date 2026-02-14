"""
Custom Security Tokenizer for Threat Intelligence SLM.

This tokenizer is specifically designed for cybersecurity domain text,
with specialized handling of:
- IP addresses (IPv4/IPv6)
- Domain names and URLs
- File hashes (MD5, SHA1, SHA256)
- CVE identifiers
- MITRE ATT&CK IDs
- Email addresses
- Registry keys
- File paths
- Threat actor names
- Malware families
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import orjson
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing


@dataclass
class SecurityVocab:
    """Security-specific vocabulary items."""
    
    # Common threat actors
    threat_actors: list[str] = field(default_factory=lambda: [
        "APT28", "APT29", "APT41", "Lazarus", "Turla", "Cozy Bear",
        "Fancy Bear", "Equation Group", "Sandworm", "Kimsuky",
        "FIN7", "FIN8", "Carbanak", "TA505", "Wizard Spider",
        "Evil Corp", "REvil", "DarkSide", "BlackCat", "LockBit",
    ])
    
    # Malware families
    malware_families: list[str] = field(default_factory=lambda: [
        "Emotet", "TrickBot", "QakBot", "IcedID", "BazarLoader",
        "Cobalt Strike", "Mimikatz", "BloodHound", "Metasploit",
        "Ryuk", "Conti", "REvil", "LockBit", "BlackCat", "Hive",
        "Agent Tesla", "FormBook", "Snake", "AsyncRAT", "NjRAT",
    ])
    
    # Attack techniques
    attack_techniques: list[str] = field(default_factory=lambda: [
        "phishing", "spearphishing", "watering_hole", "supply_chain",
        "credential_stuffing", "brute_force", "password_spray",
        "lateral_movement", "privilege_escalation", "persistence",
        "data_exfiltration", "ransomware", "cryptojacking",
        "command_and_control", "c2_beacon", "dns_tunneling",
    ])
    
    # MITRE tactics
    mitre_tactics: list[str] = field(default_factory=lambda: [
        "TA0001", "TA0002", "TA0003", "TA0004", "TA0005",
        "TA0006", "TA0007", "TA0008", "TA0009", "TA0010",
        "TA0011", "TA0040", "TA0042", "TA0043",
    ])
    
    # Security protocols and standards
    protocols: list[str] = field(default_factory=lambda: [
        "TLS", "SSL", "SSH", "RDP", "SMB", "LDAP", "Kerberos",
        "NTLM", "OAuth", "SAML", "STIX", "TAXII", "MISP",
    ])


class SecurityPatterns:
    """Regex patterns for security-specific tokens."""
    
    # IPv4 address
    IPV4 = re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    )
    
    # IPv6 address (simplified)
    IPV6 = re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|"
        r"\b(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}\b"
    )
    
    # MD5 hash
    MD5 = re.compile(r"\b[a-fA-F0-9]{32}\b")
    
    # SHA1 hash
    SHA1 = re.compile(r"\b[a-fA-F0-9]{40}\b")
    
    # SHA256 hash
    SHA256 = re.compile(r"\b[a-fA-F0-9]{64}\b")
    
    # CVE identifier
    CVE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
    
    # MITRE ATT&CK technique ID
    MITRE_TECHNIQUE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")
    
    # Domain name
    DOMAIN = re.compile(
        r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+"
        r"(?:[a-zA-Z]{2,})\b"
    )
    
    # URL
    URL = re.compile(
        r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:/[^\s]*)?"
    )
    
    # Email address
    EMAIL = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )
    
    # Windows file path
    WINDOWS_PATH = re.compile(
        r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*"
    )
    
    # Unix file path
    UNIX_PATH = re.compile(r"/(?:[^/\0]+/)*[^/\0]+")
    
    # Registry key
    REGISTRY_KEY = re.compile(
        r"\b(?:HKEY_[A-Z_]+|HK[A-Z]{2})\\[^\s]+"
    )


class SecurityTokenizer:
    """
    Custom tokenizer optimized for cybersecurity text.
    
    Features:
    - Pre-trained on security corpora
    - Special tokens for IOCs (Indicators of Compromise)
    - Subword tokenization with BPE
    - Security-specific vocabulary
    """
    
    SPECIAL_TOKENS = [
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "[IOC_IP]", "[IOC_DOMAIN]", "[IOC_HASH]", "[IOC_URL]",
        "[IOC_EMAIL]", "[IOC_CVE]", "[IOC_MITRE]", "[IOC_PATH]",
        "[THREAT_ACTOR]", "[MALWARE]", "[TECHNIQUE]", "[CAMPAIGN]",
        "[HIGH_SEV]", "[MED_SEV]", "[LOW_SEV]", "[INFO_SEV]",
    ]
    
    def __init__(
        self,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        model_path: Optional[Path] = None,
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.patterns = SecurityPatterns()
        self.vocab = SecurityVocab()
        
        if model_path and model_path.exists():
            self.tokenizer = Tokenizer.from_file(str(model_path))
        else:
            self._build_tokenizer()
    
    def _build_tokenizer(self) -> None:
        """Build the tokenizer with BPE model."""
        # Initialize BPE model
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # Normalizer chain
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
        ])
        
        # Pre-tokenizer with whitespace and punctuation
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation(),
        ])
        
        # Post-processor for BERT-style encoding
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 2),
                ("[SEP]", 3),
            ],
        )
    
    def train(self, files: list[str], output_path: Path) -> None:
        """Train the tokenizer on security corpus."""
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=True,
        )
        
        self.tokenizer.train(files, trainer)
        self.tokenizer.save(str(output_path))
    
    def _normalize_iocs(self, text: str) -> str:
        """Replace IOCs with special tokens for better generalization."""
        # Replace hashes
        text = self.patterns.SHA256.sub("[IOC_HASH]", text)
        text = self.patterns.SHA1.sub("[IOC_HASH]", text)
        text = self.patterns.MD5.sub("[IOC_HASH]", text)
        
        # Replace network IOCs
        text = self.patterns.URL.sub("[IOC_URL]", text)
        text = self.patterns.IPV4.sub("[IOC_IP]", text)
        text = self.patterns.IPV6.sub("[IOC_IP]", text)
        text = self.patterns.DOMAIN.sub("[IOC_DOMAIN]", text)
        text = self.patterns.EMAIL.sub("[IOC_EMAIL]", text)
        
        # Replace identifiers
        text = self.patterns.CVE.sub("[IOC_CVE]", text)
        text = self.patterns.MITRE_TECHNIQUE.sub("[IOC_MITRE]", text)
        
        # Replace paths
        text = self.patterns.WINDOWS_PATH.sub("[IOC_PATH]", text)
        text = self.patterns.REGISTRY_KEY.sub("[IOC_PATH]", text)
        
        return text
    
    def extract_iocs(self, text: str) -> dict[str, list[str]]:
        """Extract all IOCs from text."""
        return {
            "ipv4": self.patterns.IPV4.findall(text),
            "ipv6": self.patterns.IPV6.findall(text),
            "md5": self.patterns.MD5.findall(text),
            "sha1": self.patterns.SHA1.findall(text),
            "sha256": self.patterns.SHA256.findall(text),
            "cve": self.patterns.CVE.findall(text),
            "mitre_technique": self.patterns.MITRE_TECHNIQUE.findall(text),
            "domain": self.patterns.DOMAIN.findall(text),
            "url": self.patterns.URL.findall(text),
            "email": self.patterns.EMAIL.findall(text),
            "windows_path": self.patterns.WINDOWS_PATH.findall(text),
            "registry_key": self.patterns.REGISTRY_KEY.findall(text),
        }
    
    def encode(
        self,
        text: str,
        normalize_iocs: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
    ) -> dict:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text to encode
            normalize_iocs: Whether to replace IOCs with special tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            
        Returns:
            Dictionary with input_ids, attention_mask, and extracted IOCs
        """
        # Extract IOCs before normalization
        iocs = self.extract_iocs(text)
        
        # Optionally normalize IOCs
        if normalize_iocs:
            text = self._normalize_iocs(text)
        
        # Enable padding/truncation
        if max_length:
            self.tokenizer.enable_padding(
                pad_id=0,
                pad_token="[PAD]",
                length=max_length if padding else None,
            )
            if truncation:
                self.tokenizer.enable_truncation(max_length=max_length)
        
        # Encode
        encoding = self.tokenizer.encode(text)
        
        return {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask,
            "tokens": encoding.tokens,
            "iocs": iocs,
        }
    
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(
        self,
        texts: list[str],
        normalize_iocs: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> dict:
        """Batch encode multiple texts."""
        results = {
            "input_ids": [],
            "attention_mask": [],
            "iocs": [],
        }
        
        for text in texts:
            encoded = self.encode(
                text,
                normalize_iocs=normalize_iocs,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
            )
            results["input_ids"].append(encoded["input_ids"])
            results["attention_mask"].append(encoded["attention_mask"])
            results["iocs"].append(encoded["iocs"])
        
        return results
    
    def save(self, path: Path) -> None:
        """Save tokenizer to file."""
        self.tokenizer.save(str(path))
        
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.SPECIAL_TOKENS,
        }
        config_path = path.parent / "tokenizer_config.json"
        with open(config_path, "wb") as f:
            f.write(orjson.dumps(config))
    
    @classmethod
    def from_pretrained(cls, path: Path) -> "SecurityTokenizer":
        """Load tokenizer from file."""
        config_path = path.parent / "tokenizer_config.json"
        with open(config_path, "rb") as f:
            config = orjson.loads(f.read())
        
        tokenizer = cls(
            vocab_size=config["vocab_size"],
            min_frequency=config["min_frequency"],
            model_path=path,
        )
        return tokenizer
    
    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size."""
        return self.tokenizer.get_vocab_size()
