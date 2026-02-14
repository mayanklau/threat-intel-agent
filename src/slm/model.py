"""
Custom Small Language Model for Threat Intelligence.

This module implements a specialized transformer-based SLM optimized for:
- Threat classification and scoring
- IOC enrichment and correlation
- Attack pattern recognition
- Threat actor attribution
- Campaign clustering

Architecture: Enhanced BERT-style encoder with security-specific heads
"""

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TaskType(Enum):
    """Supported task types for the TI-SLM."""
    CLASSIFICATION = "classification"
    SEVERITY_SCORING = "severity_scoring"
    IOC_EXTRACTION = "ioc_extraction"
    THREAT_ATTRIBUTION = "threat_attribution"
    CAMPAIGN_CLUSTERING = "campaign_clustering"
    SEQUENCE_LABELING = "sequence_labeling"


@dataclass
class TISLMConfig:
    """Configuration for the Threat Intelligence SLM."""
    
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    
    # Security-specific configurations
    num_severity_classes: int = 4  # Critical, High, Medium, Low
    num_threat_types: int = 20  # Different threat categories
    num_ioc_types: int = 12  # Different IOC types
    num_mitre_tactics: int = 14  # MITRE ATT&CK tactics
    num_mitre_techniques: int = 200  # Common techniques
    
    # Training configurations
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = True
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TISLMConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def small(cls) -> "TISLMConfig":
        """Small model configuration (~30M parameters)."""
        return cls(
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
        )
    
    @classmethod
    def medium(cls) -> "TISLMConfig":
        """Medium model configuration (~110M parameters)."""
        return cls(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        )
    
    @classmethod
    def large(cls) -> "TISLMConfig":
        """Large model configuration (~340M parameters)."""
        return cls(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )


class SecurityEmbeddings(nn.Module):
    """
    Embeddings for the Security SLM.
    
    Includes:
    - Token embeddings
    - Position embeddings
    - Token type embeddings
    - IOC type embeddings (security-specific)
    """
    
    def __init__(self, config: TISLMConfig):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.ioc_type_embeddings = nn.Embedding(config.num_ioc_types + 1, config.hidden_size)  # +1 for non-IOC
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position IDs buffer
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
    
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None,
        ioc_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        if ioc_type_ids is None:
            ioc_type_ids = torch.zeros_like(input_ids)
        
        embeddings = (
            self.token_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
            + self.ioc_type_embeddings(ioc_type_ids)
        )
        
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional Flash Attention."""
    
    def __init__(self, config: TISLMConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        self.use_flash_attention = config.use_flash_attention
    
    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        if self.use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
            # Use PyTorch's native Flash Attention
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.bool()
            
            context_layer = F.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # Standard attention
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            
            context_layer = torch.matmul(attention_probs, value_layer)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)
        
        return context_layer


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward."""
    
    def __init__(self, config: TISLMConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attention_output = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_output(attention_output)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = self.attention_norm(hidden_states + attention_output)
        
        # Feed-forward
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = F.gelu(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        hidden_states = self.output_norm(hidden_states + layer_output)
        
        return hidden_states


class ThreatIntelEncoder(nn.Module):
    """Stack of transformer blocks for the TI-SLM."""
    
    def __init__(self, config: TISLMConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
    
    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)
        
        return hidden_states


class ThreatClassificationHead(nn.Module):
    """Classification head for threat type prediction."""
    
    def __init__(self, config: TISLMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_threat_types)
    
    def forward(self, pooled_output: Tensor) -> Tensor:
        x = self.dense(pooled_output)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class SeverityScoringHead(nn.Module):
    """Head for threat severity scoring."""
    
    def __init__(self, config: TISLMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_severity_classes)
    
    def forward(self, pooled_output: Tensor) -> Tensor:
        x = self.dense(pooled_output)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class MITREAttackHead(nn.Module):
    """Multi-label classification head for MITRE ATT&CK mapping."""
    
    def __init__(self, config: TISLMConfig):
        super().__init__()
        self.tactic_dense = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.tactic_classifier = nn.Linear(config.hidden_size // 2, config.num_mitre_tactics)
        
        self.technique_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.technique_classifier = nn.Linear(config.hidden_size, config.num_mitre_techniques)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, pooled_output: Tensor) -> tuple[Tensor, Tensor]:
        # Tactics (multi-label)
        tactic_hidden = F.relu(self.tactic_dense(pooled_output))
        tactic_hidden = self.dropout(tactic_hidden)
        tactic_logits = self.tactic_classifier(tactic_hidden)
        
        # Techniques (multi-label)
        technique_hidden = F.relu(self.technique_dense(pooled_output))
        technique_hidden = self.dropout(technique_hidden)
        technique_logits = self.technique_classifier(technique_hidden)
        
        return tactic_logits, technique_logits


class IOCSequenceLabelingHead(nn.Module):
    """Sequence labeling head for IOC extraction (NER-style)."""
    
    def __init__(self, config: TISLMConfig):
        super().__init__()
        # BIO tagging: B-IOC, I-IOC, O for each IOC type
        num_labels = config.num_ioc_types * 2 + 1  # B, I for each type + O
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, sequence_output: Tensor) -> Tensor:
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


class ThreatIntelSLM(nn.Module):
    """
    Threat Intelligence Small Language Model.
    
    A specialized transformer model for security operations with multiple
    task-specific heads for comprehensive threat analysis.
    """
    
    def __init__(self, config: TISLMConfig):
        super().__init__()
        self.config = config
        
        # Core model
        self.embeddings = SecurityEmbeddings(config)
        self.encoder = ThreatIntelEncoder(config)
        
        # Pooler
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Task-specific heads
        self.threat_classifier = ThreatClassificationHead(config)
        self.severity_scorer = SeverityScoringHead(config)
        self.mitre_mapper = MITREAttackHead(config)
        self.ioc_extractor = IOCSequenceLabelingHead(config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_extended_attention_mask(self, attention_mask: Tensor) -> Tensor:
        """Create extended attention mask for self-attention."""
        extended_mask = attention_mask[:, None, None, :]
        extended_mask = (1.0 - extended_mask) * torch.finfo(torch.float32).min
        return extended_mask
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        ioc_type_ids: Optional[Tensor] = None,
        task: TaskType = TaskType.CLASSIFICATION,
    ) -> dict[str, Tensor]:
        """
        Forward pass with task-specific outputs.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs [batch_size, seq_length]
            ioc_type_ids: IOC type IDs [batch_size, seq_length]
            task: Which task to perform
            
        Returns:
            Dictionary with task-specific outputs
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings
        embeddings = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            ioc_type_ids=ioc_type_ids,
        )
        
        # Extend attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        # Encode
        sequence_output = self.encoder(embeddings, extended_attention_mask)
        
        # Pool [CLS] token
        pooled_output = self.pooler(sequence_output[:, 0])
        pooled_output = torch.tanh(pooled_output)
        
        outputs = {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }
        
        # Task-specific outputs
        if task == TaskType.CLASSIFICATION:
            outputs["threat_logits"] = self.threat_classifier(pooled_output)
        
        elif task == TaskType.SEVERITY_SCORING:
            outputs["severity_logits"] = self.severity_scorer(pooled_output)
        
        elif task == TaskType.IOC_EXTRACTION:
            outputs["ioc_logits"] = self.ioc_extractor(sequence_output)
        
        elif task == TaskType.THREAT_ATTRIBUTION:
            tactic_logits, technique_logits = self.mitre_mapper(pooled_output)
            outputs["tactic_logits"] = tactic_logits
            outputs["technique_logits"] = technique_logits
        
        elif task == TaskType.CAMPAIGN_CLUSTERING:
            # Return embeddings for clustering
            outputs["embeddings"] = pooled_output
        
        return outputs
    
    def get_embeddings(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Get document embeddings for similarity/clustering."""
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=TaskType.CAMPAIGN_CLUSTERING,
        )
        return outputs["embeddings"]
    
    def save_pretrained(self, save_directory: Path) -> None:
        """Save model and config."""
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.state_dict(), save_directory / "model.pt")
        
        # Save config
        import json
        with open(save_directory / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_directory: Path) -> "ThreatIntelSLM":
        """Load model from directory."""
        import json
        
        with open(model_directory / "config.json") as f:
            config_dict = json.load(f)
        
        config = TISLMConfig.from_dict(config_dict)
        model = cls(config)
        
        state_dict = torch.load(model_directory / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
    
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class ThreatIntelInference:
    """
    Inference wrapper for the Threat Intelligence SLM.
    
    Provides easy-to-use methods for common security tasks.
    """
    
    SEVERITY_LABELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    def __init__(
        self,
        model: ThreatIntelSLM,
        tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    @torch.no_grad()
    def classify_threat(self, text: str) -> dict:
        """Classify threat type with confidence scores."""
        encoded = self.tokenizer.encode(text, max_length=512, truncation=True)
        
        input_ids = torch.tensor([encoded["input_ids"]], device=self.device)
        attention_mask = torch.tensor([encoded["attention_mask"]], device=self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=TaskType.CLASSIFICATION,
        )
        
        probs = F.softmax(outputs["threat_logits"], dim=-1)[0]
        top_k = torch.topk(probs, k=5)
        
        return {
            "predictions": [
                {"class_id": idx.item(), "confidence": prob.item()}
                for idx, prob in zip(top_k.indices, top_k.values)
            ],
            "iocs_extracted": encoded["iocs"],
        }
    
    @torch.no_grad()
    def score_severity(self, text: str) -> dict:
        """Score threat severity."""
        encoded = self.tokenizer.encode(text, max_length=512, truncation=True)
        
        input_ids = torch.tensor([encoded["input_ids"]], device=self.device)
        attention_mask = torch.tensor([encoded["attention_mask"]], device=self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=TaskType.SEVERITY_SCORING,
        )
        
        probs = F.softmax(outputs["severity_logits"], dim=-1)[0]
        predicted_class = torch.argmax(probs).item()
        
        return {
            "severity": self.SEVERITY_LABELS[predicted_class],
            "confidence": probs[predicted_class].item(),
            "all_scores": {
                label: probs[i].item() 
                for i, label in enumerate(self.SEVERITY_LABELS)
            },
        }
    
    @torch.no_grad()
    def map_to_mitre(self, text: str, threshold: float = 0.5) -> dict:
        """Map threat description to MITRE ATT&CK."""
        encoded = self.tokenizer.encode(text, max_length=512, truncation=True)
        
        input_ids = torch.tensor([encoded["input_ids"]], device=self.device)
        attention_mask = torch.tensor([encoded["attention_mask"]], device=self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=TaskType.THREAT_ATTRIBUTION,
        )
        
        tactic_probs = torch.sigmoid(outputs["tactic_logits"])[0]
        technique_probs = torch.sigmoid(outputs["technique_logits"])[0]
        
        tactics = [
            {"id": f"TA{i:04d}", "confidence": prob.item()}
            for i, prob in enumerate(tactic_probs)
            if prob > threshold
        ]
        
        techniques = [
            {"id": f"T{i:04d}", "confidence": prob.item()}
            for i, prob in enumerate(technique_probs)
            if prob > threshold
        ]
        
        return {
            "tactics": sorted(tactics, key=lambda x: -x["confidence"]),
            "techniques": sorted(techniques, key=lambda x: -x["confidence"]),
        }
    
    @torch.no_grad()
    def get_embedding(self, text: str) -> Tensor:
        """Get embedding for similarity/clustering."""
        encoded = self.tokenizer.encode(text, max_length=512, truncation=True)
        
        input_ids = torch.tensor([encoded["input_ids"]], device=self.device)
        attention_mask = torch.tensor([encoded["attention_mask"]], device=self.device)
        
        return self.model.get_embeddings(input_ids, attention_mask)[0]
    
    @torch.no_grad()
    def batch_get_embeddings(self, texts: list[str]) -> Tensor:
        """Get embeddings for multiple texts."""
        encoded = self.tokenizer.batch_encode(texts, max_length=512, truncation=True)
        
        input_ids = torch.tensor(encoded["input_ids"], device=self.device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=self.device)
        
        return self.model.get_embeddings(input_ids, attention_mask)
