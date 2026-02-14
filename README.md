# Threat Intelligence Agent

A production-ready Threat Intelligence Agent for Agentic SOC platforms, featuring a custom Small Language Model (SLM) optimized for cybersecurity and an advanced multi-tier memory system with Bead Memory for attack chain correlation.

## Features

### 🧠 Custom Security SLM
- **Security-focused tokenizer** with IOC pattern recognition (IPs, domains, hashes, CVEs, MITRE ATT&CK IDs)
- **Transformer-based architecture** (~30M to ~340M parameters)
- **Task-specific heads**:
  - Threat classification (20 categories)
  - Severity scoring (Critical/High/Medium/Low)
  - MITRE ATT&CK mapping (14 tactics, 200+ techniques)
  - IOC extraction with BIO tagging
- **Production optimizations**: Flash Attention, gradient checkpointing

### 💾 Multi-Tier Memory System
- **Working Memory**: Current investigation context (Redis-backed, TTL-based)
- **Episodic Memory**: Historical investigations (Vector DB with semantic search)
- **Semantic Memory**: Threat knowledge base (Vector DB)
- **Bead Memory**: Novel attack chain correlation system
  - Treats threat intelligence as interconnected "beads"
  - Auto-correlates across temporal and semantic dimensions
  - Forms attack chain "strings" for pattern analysis

### 🔗 Threat Feed Integrations
- AlienVault OTX
- VirusTotal
- MISP (Malware Information Sharing Platform)
- AbuseIPDB
- STIX/TAXII 2.x

### 🚀 Production Features
- FastAPI REST API with OpenAPI documentation
- Prometheus metrics and Grafana dashboards
- Docker deployment with docker-compose
- Async processing for high concurrency
- Rate limiting and caching
- Structured logging

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for full deployment)
- Redis (for working memory)
- ChromaDB (for vector storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/agentic-soc/threat-intel-agent.git
cd threat-intel-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Running with Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f threat-intel-agent

# Stop services
docker-compose down
```

### Running Locally

```bash
# Start Redis (required for working memory)
docker run -d -p 6379:6379 redis:7-alpine

# Start ChromaDB (required for vector storage)
docker run -d -p 8000:8000 chromadb/chroma

# Start the API server
ti-cli server --reload
```

## Usage

### CLI

```bash
# Enrich a single IOC
ti-cli enrich 192.168.1.100

# Enrich IOCs from file
ti-cli enrich-batch iocs.txt -o results.json

# Start threat hunt
ti-cli hunt "Potential Cobalt Strike activity"

# Start investigation
ti-cli investigate "8.8.8.8,evil.com,abc123hash" -t incident_analysis

# Analyze attack chain
ti-cli analyze-chain chain-uuid-here

# Start API server
ti-cli server --port 8080 --workers 4
```

### REST API

```bash
# Health check
curl http://localhost:8080/health

# Enrich IOC
curl -X POST http://localhost:8080/api/v1/enrich \
  -H "Content-Type: application/json" \
  -d '{"ioc": "192.168.1.100"}'

# Batch enrichment
curl -X POST http://localhost:8080/api/v1/enrich/batch \
  -H "Content-Type: application/json" \
  -d '{"iocs": ["192.168.1.100", "evil.com", "abc123"]}'

# Start investigation
curl -X POST http://localhost:8080/api/v1/investigations \
  -H "Content-Type: application/json" \
  -d '{
    "type": "incident_analysis",
    "iocs": ["8.8.8.8"],
    "description": "Investigating suspicious traffic"
  }'

# Query memory
curl -X POST http://localhost:8080/api/v1/memory/query \
  -H "Content-Type: application/json" \
  -d '{"query": "APT29 techniques", "tiers": ["semantic", "episodic"]}'
```

### Python SDK

```python
from src.agent import ThreatIntelAgent
from src.config import get_settings

async def main():
    settings = get_settings()
    
    async with ThreatIntelAgent(settings) as agent:
        # Enrich an IOC
        result = await agent.enrich_ioc("192.168.1.100")
        print(f"Threat Score: {result.threat_score}")
        print(f"Classification: {result.classification}")
        print(f"MITRE Techniques: {result.mitre_techniques}")
        
        # Start investigation
        investigation = await agent.start_investigation(
            investigation_type="incident_analysis",
            initial_iocs=["8.8.8.8", "evil.com"],
            description="Investigating C2 traffic"
        )
        
        # Threat hunting
        results = await agent.hunt_threats(
            hypothesis="Lateral movement via RDP",
            time_range_hours=72
        )
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         REST API (FastAPI)                       │
├─────────────────────────────────────────────────────────────────┤
│                     Threat Intel Agent Core                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  IOC Enrich  │  │ Threat Hunt  │  │  Attack Chain Anlyz  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Custom Security SLM                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Tokenizer   │  │  Transformer │  │    Task Heads        │  │
│  │  (IOC-aware) │  │   Encoder    │  │ (Class/Sev/MITRE)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                  Multi-Tier Memory System                        │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ Working │ │ Episodic │ │ Semantic │ │   Bead Memory    │   │
│  │ (Redis) │ │(ChromaDB)│ │(ChromaDB)│ │ (Graph+Vector)   │   │
│  └─────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   Threat Feed Integrations                       │
│  ┌─────┐  ┌────────────┐  ┌──────┐  ┌──────────┐  ┌───────┐   │
│  │ OTX │  │ VirusTotal │  │ MISP │  │ AbuseIPDB│  │ TAXII │   │
│  └─────┘  └────────────┘  └──────┘  └──────────┘  └───────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

See `.env.example` for all configuration options. Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `SLM_MODEL_SIZE` | Model size (small/medium/large) | medium |
| `MEMORY_REDIS_URL` | Redis connection URL | redis://localhost:6379/0 |
| `MEMORY_CHROMA_HOST` | ChromaDB host | localhost |
| `THREATFEED_OTX_API_KEY` | AlienVault OTX API key | - |
| `AGENT_AUTO_CORRELATE` | Enable auto-correlation | true |

### YAML Configuration

```yaml
# config.yml
environment: production
debug: false

slm:
  model_size: medium
  use_flash_attention: true

memory:
  bead_temporal_window: 604800  # 7 days
  bead_similarity_threshold: 0.7

agent:
  enrichment_providers:
    - otx
    - virustotal
    - abuseipdb
  auto_correlate: true
```

## Bead Memory System

The Bead Memory system is a novel approach to threat intelligence correlation:

```python
# A Bead represents a single piece of threat intelligence
bead = Bead(
    id="bead-123",
    bead_type="ioc",  # ioc, technique, actor, campaign, incident, artifact
    content="192.168.1.100",
    embedding=[...],  # Vector representation
    timestamp=datetime.now(),
    metadata={"severity": "high", "source": "firewall"}
)

# BeadStrings chain related beads into attack narratives
string = BeadString(
    id="string-456",
    beads=[bead1, bead2, bead3],  # Chronologically ordered
    confidence=0.85,
    campaign="APT29-2024"
)
```

### Auto-Correlation

The system automatically correlates beads based on:
- **Temporal proximity**: Events within configurable time windows
- **Semantic similarity**: Vector similarity above threshold
- **Shared attributes**: Common campaigns, actors, techniques

## Model Training

The SLM requires training on security-specific corpus:

```bash
# Initialize model for training
ti-cli train --data ./training-data --output ./models/trained --epochs 10

# Training data should include:
# - Threat reports and advisories
# - CVE descriptions
# - MITRE ATT&CK technique descriptions
# - Malware analysis reports
# - Labeled IOC datasets
```

## API Documentation

Once running, access the API documentation:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc
- OpenAPI JSON: http://localhost:8080/openapi.json

## Monitoring

### Prometheus Metrics

Available at `/metrics/prometheus`:
- `ti_requests_total`: Total API requests
- `ti_request_latency_seconds`: Request latency histogram
- `ti_ioc_enrichments_total`: IOC enrichments by type
- `ti_memory_operations_total`: Memory operations by tier
- `ti_active_investigations`: Current active investigations

### Grafana Dashboards

Pre-configured dashboards available at http://localhost:3000:
- Threat Intel Overview
- IOC Enrichment Performance
- Memory System Health
- Attack Chain Analytics

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Code formatting
black src tests
ruff src tests

# Type checking
mypy src
```

## Project Structure

```
threat-intel-agent/
├── src/
│   ├── agent/           # Agent core and orchestration
│   │   └── core.py
│   ├── api/             # FastAPI REST endpoints
│   │   └── main.py
│   ├── integrations/    # Threat feed providers
│   │   └── threat_feeds.py
│   ├── memory/          # Multi-tier memory system
│   │   └── memory_system.py
│   ├── slm/             # Custom language model
│   │   ├── model.py
│   │   └── tokenizer.py
│   ├── cli.py           # CLI interface
│   └── config.py        # Configuration management
├── tests/
├── configs/
├── data/
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## License

MIT License - See LICENSE for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## Acknowledgments

- MITRE ATT&CK Framework
- STIX/TAXII Standards
- Open Threat Intelligence Community
