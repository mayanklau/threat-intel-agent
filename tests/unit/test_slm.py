"""
Unit tests for the Security Tokenizer and Threat Intel SLM.
"""

import pytest
import torch


class TestSecurityTokenizer:
    """Tests for the SecurityTokenizer class."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initializes correctly."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        assert tokenizer is not None
        assert tokenizer.vocab_size > 0
    
    def test_ioc_pattern_detection_ipv4(self):
        """Test IPv4 address detection."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        text = "The malicious IP 192.168.1.100 was observed"
        iocs = tokenizer.extract_iocs(text)
        
        assert any(ioc["type"] == "ipv4" and ioc["value"] == "192.168.1.100" for ioc in iocs)
    
    def test_ioc_pattern_detection_domain(self):
        """Test domain detection."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        text = "Traffic to evil-malware.com was blocked"
        iocs = tokenizer.extract_iocs(text)
        
        assert any(ioc["type"] == "domain" for ioc in iocs)
    
    def test_ioc_pattern_detection_hash_md5(self):
        """Test MD5 hash detection."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        text = "File hash: d41d8cd98f00b204e9800998ecf8427e"
        iocs = tokenizer.extract_iocs(text)
        
        assert any(ioc["type"] == "md5" for ioc in iocs)
    
    def test_ioc_pattern_detection_hash_sha256(self):
        """Test SHA256 hash detection."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        text = "SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        iocs = tokenizer.extract_iocs(text)
        
        assert any(ioc["type"] == "sha256" for ioc in iocs)
    
    def test_ioc_pattern_detection_cve(self):
        """Test CVE ID detection."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        text = "Exploiting CVE-2024-1234 vulnerability"
        iocs = tokenizer.extract_iocs(text)
        
        assert any(ioc["type"] == "cve" and "CVE-2024-1234" in ioc["value"] for ioc in iocs)
    
    def test_ioc_pattern_detection_mitre(self):
        """Test MITRE ATT&CK ID detection."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        text = "Using technique T1059.001 for execution"
        iocs = tokenizer.extract_iocs(text)
        
        assert any(ioc["type"] == "mitre" for ioc in iocs)
    
    def test_encode_text(self):
        """Test text encoding."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        text = "Malware detected on host 192.168.1.100"
        
        encoded = tokenizer.encode(text)
        
        assert isinstance(encoded, list)
        assert len(encoded) > 0
        assert all(isinstance(t, int) for t in encoded)
    
    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip preserves meaning."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        text = "APT29 used PowerShell for initial access"
        
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        # Should preserve key terms
        assert "APT29" in decoded or "apt" in decoded.lower()
        assert "PowerShell" in decoded or "powershell" in decoded.lower()
    
    def test_batch_encoding(self):
        """Test batch encoding of multiple texts."""
        from src.slm.tokenizer import SecurityTokenizer
        
        tokenizer = SecurityTokenizer()
        texts = [
            "First threat report",
            "Second threat report with IP 10.0.0.1",
            "Third report about CVE-2024-5678",
        ]
        
        encoded = tokenizer.encode_batch(texts, padding=True)
        
        assert len(encoded) == 3
        # All should be same length when padded
        lengths = [len(e) for e in encoded]
        assert len(set(lengths)) == 1


class TestThreatIntelSLM:
    """Tests for the Threat Intelligence SLM."""
    
    def test_model_initialization_small(self):
        """Test small model initializes correctly."""
        from src.slm.model import ThreatIntelSLM, get_model_config
        
        config = get_model_config("small")
        model = ThreatIntelSLM(config)
        
        assert model is not None
        # Small model should have fewer parameters
        params = sum(p.numel() for p in model.parameters())
        assert params < 50_000_000  # Less than 50M
    
    def test_model_initialization_medium(self):
        """Test medium model initializes correctly."""
        from src.slm.model import ThreatIntelSLM, get_model_config
        
        config = get_model_config("medium")
        model = ThreatIntelSLM(config)
        
        assert model is not None
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        from src.slm.model import ThreatIntelSLM, get_model_config
        
        config = get_model_config("small")
        model = ThreatIntelSLM(config)
        model.eval()
        
        # Create dummy input
        batch_size = 2
        seq_length = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        assert "last_hidden_state" in outputs
        assert outputs["last_hidden_state"].shape == (batch_size, seq_length, config.hidden_size)
    
    def test_threat_classification_head(self):
        """Test threat classification head output."""
        from src.slm.model import ThreatIntelSLM, get_model_config
        
        config = get_model_config("small")
        model = ThreatIntelSLM(config)
        model.eval()
        
        batch_size = 2
        seq_length = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        if "threat_logits" in outputs:
            assert outputs["threat_logits"].shape[0] == batch_size
            assert outputs["threat_logits"].shape[1] == 20  # 20 threat categories
    
    def test_severity_scoring_head(self):
        """Test severity scoring head output."""
        from src.slm.model import ThreatIntelSLM, get_model_config
        
        config = get_model_config("small")
        model = ThreatIntelSLM(config)
        model.eval()
        
        batch_size = 2
        seq_length = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        if "severity_logits" in outputs:
            assert outputs["severity_logits"].shape[0] == batch_size
            assert outputs["severity_logits"].shape[1] == 4  # Critical/High/Medium/Low
    
    def test_model_save_load(self, tmp_path):
        """Test model save and load."""
        from src.slm.model import ThreatIntelSLM, get_model_config
        
        config = get_model_config("small")
        model = ThreatIntelSLM(config)
        
        # Save model
        save_path = tmp_path / "model"
        model.save_pretrained(save_path)
        
        # Load model
        loaded_model = ThreatIntelSLM.from_pretrained(save_path)
        
        # Verify weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)


class TestThreatIntelInference:
    """Tests for the ThreatIntelInference wrapper."""
    
    def test_inference_initialization(self):
        """Test inference wrapper initializes."""
        from src.slm.model import ThreatIntelInference, get_model_config
        from src.slm.tokenizer import SecurityTokenizer
        
        config = get_model_config("small")
        tokenizer = SecurityTokenizer()
        
        inference = ThreatIntelInference(config, tokenizer)
        
        assert inference is not None
    
    def test_classify_threat(self):
        """Test threat classification."""
        from src.slm.model import ThreatIntelInference, get_model_config
        from src.slm.tokenizer import SecurityTokenizer
        
        config = get_model_config("small")
        tokenizer = SecurityTokenizer()
        inference = ThreatIntelInference(config, tokenizer)
        
        text = "Detected Cobalt Strike beacon on compromised host"
        result = inference.classify_threat(text)
        
        assert "label" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1
    
    def test_score_severity(self):
        """Test severity scoring."""
        from src.slm.model import ThreatIntelInference, get_model_config
        from src.slm.tokenizer import SecurityTokenizer
        
        config = get_model_config("small")
        tokenizer = SecurityTokenizer()
        inference = ThreatIntelInference(config, tokenizer)
        
        text = "Critical vulnerability allows remote code execution"
        result = inference.score_severity(text)
        
        assert "severity" in result
        assert result["severity"] in ["critical", "high", "medium", "low"]
        assert "confidence" in result
    
    def test_map_to_mitre(self):
        """Test MITRE ATT&CK mapping."""
        from src.slm.model import ThreatIntelInference, get_model_config
        from src.slm.tokenizer import SecurityTokenizer
        
        config = get_model_config("small")
        tokenizer = SecurityTokenizer()
        inference = ThreatIntelInference(config, tokenizer)
        
        text = "Attacker used PowerShell to download and execute malicious payload"
        result = inference.map_to_mitre(text)
        
        assert "tactics" in result
        assert "techniques" in result
        assert isinstance(result["techniques"], list)
    
    def test_get_embedding(self):
        """Test embedding generation."""
        from src.slm.model import ThreatIntelInference, get_model_config
        from src.slm.tokenizer import SecurityTokenizer
        
        config = get_model_config("small")
        tokenizer = SecurityTokenizer()
        inference = ThreatIntelInference(config, tokenizer)
        
        text = "APT29 targeting government organizations"
        embedding = inference.get_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == config.hidden_size
        assert all(isinstance(v, float) for v in embedding)
