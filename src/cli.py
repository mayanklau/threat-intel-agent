#!/usr/bin/env python3
"""
Command Line Interface for Threat Intelligence Agent.

Usage:
    ti-cli enrich <ioc>             Enrich a single IOC
    ti-cli enrich-batch <file>      Enrich IOCs from file
    ti-cli hunt <hypothesis>        Start threat hunt
    ti-cli investigate <iocs>       Start investigation
    ti-cli analyze-chain <id>       Analyze attack chain
    ti-cli server                   Start API server
    ti-cli train                    Train/fine-tune model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)


def print_json(data: Any) -> None:
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2, default=str))


def print_result(title: str, data: Any) -> None:
    """Print formatted result."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print_json(data)


async def cmd_enrich(args: argparse.Namespace) -> int:
    """Enrich a single IOC."""
    from src.agent import ThreatIntelAgent
    from src.config import get_settings
    
    settings = get_settings()
    
    print(f"Enriching IOC: {args.ioc}")
    
    async with ThreatIntelAgent(settings) as agent:
        result = await agent.enrich_ioc(args.ioc)
        
        print_result("Enrichment Result", {
            "ioc": result.ioc,
            "ioc_type": result.ioc_type,
            "classification": result.classification,
            "severity": result.severity,
            "confidence": result.confidence,
            "threat_score": result.threat_score,
            "mitre_techniques": result.mitre_techniques[:5] if result.mitre_techniques else [],
            "related_campaigns": result.related_campaigns,
            "sources": list(result.provider_results.keys()),
        })
    
    return 0


async def cmd_enrich_batch(args: argparse.Namespace) -> int:
    """Enrich IOCs from file."""
    from src.agent import ThreatIntelAgent
    from src.config import get_settings
    
    settings = get_settings()
    
    # Read IOCs from file
    ioc_file = Path(args.file)
    if not ioc_file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1
    
    iocs = [line.strip() for line in ioc_file.read_text().splitlines() if line.strip()]
    print(f"Enriching {len(iocs)} IOCs from {args.file}")
    
    async with ThreatIntelAgent(settings) as agent:
        results = []
        for i, ioc in enumerate(iocs, 1):
            print(f"  [{i}/{len(iocs)}] Processing: {ioc}")
            try:
                result = await agent.enrich_ioc(ioc)
                results.append({
                    "ioc": result.ioc,
                    "type": result.ioc_type,
                    "severity": result.severity,
                    "score": result.threat_score,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "ioc": ioc,
                    "status": "error",
                    "error": str(e)
                })
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(results, indent=2))
            print(f"\nResults saved to: {args.output}")
        else:
            print_result("Batch Enrichment Results", results)
    
    return 0


async def cmd_hunt(args: argparse.Namespace) -> int:
    """Start a threat hunt."""
    from src.agent import ThreatIntelAgent
    from src.config import get_settings
    
    settings = get_settings()
    
    print(f"Starting threat hunt with hypothesis: {args.hypothesis}")
    
    async with ThreatIntelAgent(settings) as agent:
        results = await agent.hunt_threats(
            hypothesis=args.hypothesis,
            time_range_hours=args.hours,
        )
        
        print_result("Threat Hunt Results", {
            "hypothesis": args.hypothesis,
            "findings_count": len(results.get("findings", [])),
            "findings": results.get("findings", [])[:10],
            "related_iocs": results.get("related_iocs", [])[:10],
            "recommendations": results.get("recommendations", []),
        })
    
    return 0


async def cmd_investigate(args: argparse.Namespace) -> int:
    """Start an investigation."""
    from src.agent import ThreatIntelAgent, InvestigationType
    from src.config import get_settings
    
    settings = get_settings()
    iocs = [ioc.strip() for ioc in args.iocs.split(",")]
    
    print(f"Starting investigation with {len(iocs)} IOCs")
    
    async with ThreatIntelAgent(settings) as agent:
        investigation = await agent.start_investigation(
            investigation_type=InvestigationType(args.type),
            initial_iocs=iocs,
            description=args.description or f"Investigation of {len(iocs)} IOCs",
        )
        
        print_result("Investigation Started", {
            "investigation_id": investigation.id,
            "type": investigation.type.value,
            "status": investigation.status,
            "iocs_processed": len(investigation.processed_iocs),
            "findings": len(investigation.findings),
            "campaigns": investigation.campaigns,
            "actors": investigation.actors,
        })
    
    return 0


async def cmd_analyze_chain(args: argparse.Namespace) -> int:
    """Analyze an attack chain."""
    from src.agent import ThreatIntelAgent
    from src.config import get_settings
    
    settings = get_settings()
    
    print(f"Analyzing attack chain: {args.chain_id}")
    
    async with ThreatIntelAgent(settings) as agent:
        analysis = await agent.analyze_attack_chain(args.chain_id)
        
        print_result("Attack Chain Analysis", {
            "chain_id": args.chain_id,
            "timeline": analysis.get("timeline", [])[:10],
            "techniques": analysis.get("techniques", []),
            "attribution": analysis.get("attribution", {}),
            "impact": analysis.get("impact", {}),
            "recommendations": analysis.get("recommendations", []),
        })
    
    return 0


def cmd_server(args: argparse.Namespace) -> int:
    """Start the API server."""
    import uvicorn
    from src.config import get_settings
    
    settings = get_settings()
    
    print(f"Starting API server on {settings.api.host}:{settings.api.port}")
    
    uvicorn.run(
        "src.api.main:app",
        host=args.host or settings.api.host,
        port=args.port or settings.api.port,
        workers=args.workers or settings.api.workers,
        reload=args.reload,
        log_level=settings.api.log_level.lower(),
    )
    
    return 0


async def cmd_train(args: argparse.Namespace) -> int:
    """Train or fine-tune the model."""
    from src.slm import ThreatIntelSLM, ThreatIntelConfig, SecurityTokenizer
    from src.config import get_settings
    
    settings = get_settings()
    
    print("Training configuration:")
    print(f"  Model size: {settings.slm.model_size}")
    print(f"  Data path: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    # Initialize tokenizer
    tokenizer = SecurityTokenizer()
    
    # Initialize model
    config = ThreatIntelConfig(
        vocab_size=settings.slm.vocab_size,
        hidden_size=settings.slm.hidden_size,
        num_hidden_layers=settings.slm.num_hidden_layers,
        num_attention_heads=settings.slm.num_attention_heads,
        intermediate_size=settings.slm.intermediate_size,
    )
    model = ThreatIntelSLM(config)
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("\nNote: Full training implementation requires:")
    print("  1. Security corpus (threat reports, CVE descriptions, etc.)")
    print("  2. Labeled data for classification tasks")
    print("  3. GPU resources for efficient training")
    print("\nPlease implement the training loop based on your specific requirements.")
    
    # Save initial model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save(output_path / "tokenizer")
    
    print(f"\nInitial model saved to: {output_path}")
    
    return 0


def cmd_memory_stats(args: argparse.Namespace) -> int:
    """Display memory system statistics."""
    asyncio.run(_memory_stats(args))
    return 0


async def _memory_stats(args: argparse.Namespace) -> None:
    from src.memory import ThreatIntelMemorySystem
    from src.config import get_settings
    
    settings = get_settings()
    
    memory = ThreatIntelMemorySystem(settings.memory)
    await memory.initialize()
    
    stats = await memory.get_stats()
    
    print_result("Memory System Statistics", stats)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="ti-cli",
        description="Threat Intelligence Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Enrich command
    enrich_parser = subparsers.add_parser("enrich", help="Enrich a single IOC")
    enrich_parser.add_argument("ioc", help="IOC to enrich (IP, domain, hash, etc.)")
    
    # Enrich batch command
    batch_parser = subparsers.add_parser("enrich-batch", help="Enrich IOCs from file")
    batch_parser.add_argument("file", help="File containing IOCs (one per line)")
    batch_parser.add_argument("--output", "-o", help="Output file for results")
    
    # Hunt command
    hunt_parser = subparsers.add_parser("hunt", help="Start threat hunt")
    hunt_parser.add_argument("hypothesis", help="Threat hunting hypothesis")
    hunt_parser.add_argument("--hours", type=int, default=24, help="Time range in hours")
    
    # Investigate command
    investigate_parser = subparsers.add_parser("investigate", help="Start investigation")
    investigate_parser.add_argument("iocs", help="Comma-separated list of IOCs")
    investigate_parser.add_argument(
        "--type", "-t",
        default="ioc_enrichment",
        choices=["ioc_enrichment", "threat_hunt", "incident_analysis", 
                 "campaign_tracking", "attribution", "vulnerability_assessment"],
        help="Investigation type"
    )
    investigate_parser.add_argument("--description", "-d", help="Investigation description")
    
    # Analyze chain command
    chain_parser = subparsers.add_parser("analyze-chain", help="Analyze attack chain")
    chain_parser.add_argument("chain_id", help="Attack chain ID")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", help="Host to bind")
    server_parser.add_argument("--port", type=int, help="Port to bind")
    server_parser.add_argument("--workers", type=int, help="Number of workers")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train/fine-tune model")
    train_parser.add_argument("--data", "-d", required=True, help="Training data path")
    train_parser.add_argument("--output", "-o", default="data/models/trained", help="Output path")
    train_parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    # Memory stats command
    subparsers.add_parser("memory-stats", help="Display memory statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Load config if specified
    if args.config:
        from src.config import Settings, _settings
        import src.config
        src.config._settings = Settings.from_yaml(args.config)
    
    # Execute command
    try:
        if args.command == "enrich":
            return asyncio.run(cmd_enrich(args))
        elif args.command == "enrich-batch":
            return asyncio.run(cmd_enrich_batch(args))
        elif args.command == "hunt":
            return asyncio.run(cmd_hunt(args))
        elif args.command == "investigate":
            return asyncio.run(cmd_investigate(args))
        elif args.command == "analyze-chain":
            return asyncio.run(cmd_analyze_chain(args))
        elif args.command == "server":
            return cmd_server(args)
        elif args.command == "train":
            return asyncio.run(cmd_train(args))
        elif args.command == "memory-stats":
            return cmd_memory_stats(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        logger.exception("Command failed", error=str(e))
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
