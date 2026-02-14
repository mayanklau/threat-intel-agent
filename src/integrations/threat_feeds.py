"""
Threat Intelligence Feed Integrations.

This module provides integrations with major threat intelligence platforms:
- AlienVault OTX
- MISP
- VirusTotal
- AbuseIPDB
- Shodan
- STIX/TAXII feeds

Each provider implements the ThreatIntelProvider interface for
consistent IOC enrichment across platforms.
"""

import asyncio
import hashlib
from abc import ABC
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
import structlog
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agent.core import ThreatIntelProvider

logger = structlog.get_logger()


class CachedProvider(ThreatIntelProvider, ABC):
    """Base class for providers with caching support."""
    
    def __init__(self, cache_ttl: int = 3600, cache_maxsize: int = 10000):
        self._cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        self._lock = asyncio.Lock()
    
    def _cache_key(self, method: str, *args) -> str:
        """Generate cache key."""
        content = f"{self.name}:{method}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_cached(self, key: str) -> Optional[dict]:
        """Get cached result."""
        return self._cache.get(key)
    
    async def _set_cached(self, key: str, value: dict) -> None:
        """Set cached result."""
        async with self._lock:
            self._cache[key] = value


class OTXProvider(CachedProvider):
    """
    AlienVault Open Threat Exchange (OTX) integration.
    
    OTX provides community-sourced threat intelligence including:
    - IP reputation
    - Domain analysis
    - File hash lookups
    - Pulse (threat report) data
    """
    
    BASE_URL = "https://otx.alienvault.com/api/v1"
    
    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 3600,
        timeout: float = 30.0,
    ):
        super().__init__(cache_ttl=cache_ttl)
        self.api_key = api_key
        self.timeout = timeout
        self._client = None
        self.logger = logger.bind(provider="otx")
    
    @property
    def name(self) -> str:
        return "otx"
    
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={"X-OTX-API-KEY": self.api_key},
                timeout=self.timeout,
            )
        return self._client
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _request(self, endpoint: str) -> dict:
        """Make API request with retry."""
        client = self._get_client()
        response = await client.get(endpoint)
        response.raise_for_status()
        return response.json()
    
    async def enrich_ip(self, ip: str) -> dict:
        """Enrich IP address with OTX data."""
        cache_key = self._cache_key("ip", ip)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            # Get general info
            general = await self._request(f"/indicators/IPv4/{ip}/general")
            
            # Get reputation
            reputation = await self._request(f"/indicators/IPv4/{ip}/reputation")
            
            # Get malware samples
            malware = await self._request(f"/indicators/IPv4/{ip}/malware")
            
            # Get passive DNS
            passive_dns = await self._request(f"/indicators/IPv4/{ip}/passive_dns")
            
            result = {
                "ip": ip,
                "pulse_count": general.get("pulse_info", {}).get("count", 0),
                "reputation": reputation.get("reputation", 0),
                "malware_samples": len(malware.get("data", [])),
                "passive_dns_count": len(passive_dns.get("passive_dns", [])),
                "country": general.get("country_name"),
                "asn": general.get("asn"),
                "pulses": [
                    {
                        "id": p.get("id"),
                        "name": p.get("name"),
                        "created": p.get("created"),
                        "tags": p.get("tags", []),
                    }
                    for p in general.get("pulse_info", {}).get("pulses", [])[:10]
                ],
                "malicious": general.get("pulse_info", {}).get("count", 0) > 0,
            }
            
            await self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error("otx_ip_enrichment_failed", ip=ip, error=str(e))
            return {"ip": ip, "error": str(e)}
    
    async def enrich_domain(self, domain: str) -> dict:
        """Enrich domain with OTX data."""
        cache_key = self._cache_key("domain", domain)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            general = await self._request(f"/indicators/domain/{domain}/general")
            malware = await self._request(f"/indicators/domain/{domain}/malware")
            
            result = {
                "domain": domain,
                "pulse_count": general.get("pulse_info", {}).get("count", 0),
                "malware_samples": len(malware.get("data", [])),
                "whois": general.get("whois"),
                "pulses": [
                    {
                        "id": p.get("id"),
                        "name": p.get("name"),
                        "tags": p.get("tags", []),
                    }
                    for p in general.get("pulse_info", {}).get("pulses", [])[:10]
                ],
                "malicious": general.get("pulse_info", {}).get("count", 0) > 0,
            }
            
            await self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error("otx_domain_enrichment_failed", domain=domain, error=str(e))
            return {"domain": domain, "error": str(e)}
    
    async def enrich_hash(self, hash_value: str, hash_type: str) -> dict:
        """Enrich file hash with OTX data."""
        cache_key = self._cache_key("hash", hash_value)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            endpoint_type = hash_type.upper()
            general = await self._request(f"/indicators/file/{hash_value}/general")
            analysis = await self._request(f"/indicators/file/{hash_value}/analysis")
            
            result = {
                "hash": hash_value,
                "hash_type": hash_type,
                "pulse_count": general.get("pulse_info", {}).get("count", 0),
                "file_type": general.get("type"),
                "size": general.get("size"),
                "analysis": analysis.get("analysis", {}),
                "malicious": general.get("pulse_info", {}).get("count", 0) > 0,
            }
            
            await self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error("otx_hash_enrichment_failed", hash=hash_value, error=str(e))
            return {"hash": hash_value, "error": str(e)}
    
    async def search_ioc(self, ioc: str) -> list[dict]:
        """Search for IOC in pulses."""
        try:
            result = await self._request(f"/search/pulses?q={ioc}")
            return result.get("results", [])
        except Exception as e:
            self.logger.error("otx_search_failed", ioc=ioc, error=str(e))
            return []
    
    async def get_pulses(
        self,
        modified_since: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get recent threat pulses."""
        endpoint = f"/pulses/subscribed?limit={limit}"
        if modified_since:
            endpoint += f"&modified_since={modified_since.isoformat()}"
        
        try:
            result = await self._request(endpoint)
            return result.get("results", [])
        except Exception as e:
            self.logger.error("otx_pulses_failed", error=str(e))
            return []


class VirusTotalProvider(CachedProvider):
    """
    VirusTotal integration for file and URL analysis.
    
    Provides:
    - Multi-AV scan results
    - Behavioral analysis
    - Network indicators
    - File relationships
    """
    
    BASE_URL = "https://www.virustotal.com/api/v3"
    
    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 3600,
        timeout: float = 60.0,
    ):
        super().__init__(cache_ttl=cache_ttl)
        self.api_key = api_key
        self.timeout = timeout
        self._client = None
        self.logger = logger.bind(provider="virustotal")
    
    @property
    def name(self) -> str:
        return "virustotal"
    
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={"x-apikey": self.api_key},
                timeout=self.timeout,
            )
        return self._client
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _request(self, endpoint: str) -> dict:
        """Make API request with retry."""
        client = self._get_client()
        response = await client.get(endpoint)
        response.raise_for_status()
        return response.json()
    
    async def enrich_ip(self, ip: str) -> dict:
        """Enrich IP with VirusTotal data."""
        cache_key = self._cache_key("ip", ip)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            data = await self._request(f"/ip_addresses/{ip}")
            attributes = data.get("data", {}).get("attributes", {})
            stats = attributes.get("last_analysis_stats", {})
            
            result = {
                "ip": ip,
                "country": attributes.get("country"),
                "asn": attributes.get("asn"),
                "as_owner": attributes.get("as_owner"),
                "malicious_count": stats.get("malicious", 0),
                "suspicious_count": stats.get("suspicious", 0),
                "harmless_count": stats.get("harmless", 0),
                "undetected_count": stats.get("undetected", 0),
                "reputation": attributes.get("reputation", 0),
                "malicious": stats.get("malicious", 0) > 0,
                "suspicious": stats.get("suspicious", 0) > 0,
            }
            
            await self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error("vt_ip_enrichment_failed", ip=ip, error=str(e))
            return {"ip": ip, "error": str(e)}
    
    async def enrich_domain(self, domain: str) -> dict:
        """Enrich domain with VirusTotal data."""
        cache_key = self._cache_key("domain", domain)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            data = await self._request(f"/domains/{domain}")
            attributes = data.get("data", {}).get("attributes", {})
            stats = attributes.get("last_analysis_stats", {})
            
            result = {
                "domain": domain,
                "registrar": attributes.get("registrar"),
                "creation_date": attributes.get("creation_date"),
                "malicious_count": stats.get("malicious", 0),
                "suspicious_count": stats.get("suspicious", 0),
                "harmless_count": stats.get("harmless", 0),
                "reputation": attributes.get("reputation", 0),
                "categories": attributes.get("categories", {}),
                "malicious": stats.get("malicious", 0) > 0,
                "suspicious": stats.get("suspicious", 0) > 0,
            }
            
            await self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error("vt_domain_enrichment_failed", domain=domain, error=str(e))
            return {"domain": domain, "error": str(e)}
    
    async def enrich_hash(self, hash_value: str, hash_type: str) -> dict:
        """Enrich file hash with VirusTotal data."""
        cache_key = self._cache_key("hash", hash_value)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            data = await self._request(f"/files/{hash_value}")
            attributes = data.get("data", {}).get("attributes", {})
            stats = attributes.get("last_analysis_stats", {})
            
            result = {
                "hash": hash_value,
                "hash_type": hash_type,
                "file_type": attributes.get("type_description"),
                "size": attributes.get("size"),
                "names": attributes.get("names", [])[:10],
                "malicious_count": stats.get("malicious", 0),
                "suspicious_count": stats.get("suspicious", 0),
                "harmless_count": stats.get("harmless", 0),
                "undetected_count": stats.get("undetected", 0),
                "first_submission": attributes.get("first_submission_date"),
                "last_submission": attributes.get("last_submission_date"),
                "times_submitted": attributes.get("times_submitted"),
                "tags": attributes.get("tags", []),
                "malicious": stats.get("malicious", 0) > 0,
                "suspicious": stats.get("suspicious", 0) > 0,
            }
            
            await self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error("vt_hash_enrichment_failed", hash=hash_value, error=str(e))
            return {"hash": hash_value, "error": str(e)}
    
    async def search_ioc(self, ioc: str) -> list[dict]:
        """Search VirusTotal for IOC."""
        try:
            data = await self._request(f"/search?query={ioc}")
            return data.get("data", [])
        except Exception as e:
            self.logger.error("vt_search_failed", ioc=ioc, error=str(e))
            return []


class MISPProvider(CachedProvider):
    """
    MISP (Malware Information Sharing Platform) integration.
    
    MISP provides:
    - Structured threat intelligence
    - IOC correlation
    - Event-based threat sharing
    - Taxonomies and galaxies
    """
    
    def __init__(
        self,
        url: str,
        api_key: str,
        verify_ssl: bool = True,
        cache_ttl: int = 1800,
        timeout: float = 30.0,
    ):
        super().__init__(cache_ttl=cache_ttl)
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self._client = None
        self.logger = logger.bind(provider="misp")
    
    @property
    def name(self) -> str:
        return "misp"
    
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.url,
                headers={
                    "Authorization": self.api_key,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
        return self._client
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[dict] = None,
    ) -> dict:
        """Make API request with retry."""
        client = self._get_client()
        
        if method == "GET":
            response = await client.get(endpoint)
        else:
            response = await client.post(endpoint, json=json)
        
        response.raise_for_status()
        return response.json()
    
    async def enrich_ip(self, ip: str) -> dict:
        """Search MISP for IP."""
        return await self._search_attribute("ip-src", ip) or await self._search_attribute("ip-dst", ip)
    
    async def enrich_domain(self, domain: str) -> dict:
        """Search MISP for domain."""
        return await self._search_attribute("domain", domain)
    
    async def enrich_hash(self, hash_value: str, hash_type: str) -> dict:
        """Search MISP for file hash."""
        type_mapping = {
            "md5": "md5",
            "sha1": "sha1",
            "sha256": "sha256",
        }
        misp_type = type_mapping.get(hash_type.lower(), hash_type)
        return await self._search_attribute(misp_type, hash_value)
    
    async def _search_attribute(self, attr_type: str, value: str) -> dict:
        """Search for attribute in MISP."""
        cache_key = self._cache_key("attr", attr_type, value)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            result = await self._request(
                "POST",
                "/attributes/restSearch",
                json={
                    "type": attr_type,
                    "value": value,
                    "includeEventTags": True,
                    "limit": 100,
                },
            )
            
            attributes = result.get("response", {}).get("Attribute", [])
            
            # Extract events
            events = {}
            for attr in attributes:
                event_id = attr.get("event_id")
                if event_id and event_id not in events:
                    events[event_id] = {
                        "id": event_id,
                        "info": attr.get("Event", {}).get("info"),
                        "threat_level": attr.get("Event", {}).get("threat_level_id"),
                        "tags": [t.get("name") for t in attr.get("Tag", [])],
                    }
            
            response = {
                "type": attr_type,
                "value": value,
                "attribute_count": len(attributes),
                "events": list(events.values()),
                "malicious": len(attributes) > 0,
            }
            
            await self._set_cached(cache_key, response)
            return response
            
        except Exception as e:
            self.logger.error("misp_search_failed", type=attr_type, value=value, error=str(e))
            return {"type": attr_type, "value": value, "error": str(e)}
    
    async def search_ioc(self, ioc: str) -> list[dict]:
        """Search MISP for any IOC type."""
        try:
            result = await self._request(
                "POST",
                "/attributes/restSearch",
                json={
                    "value": ioc,
                    "limit": 50,
                },
            )
            return result.get("response", {}).get("Attribute", [])
        except Exception as e:
            self.logger.error("misp_ioc_search_failed", ioc=ioc, error=str(e))
            return []
    
    async def get_events(
        self,
        days: int = 7,
        limit: int = 100,
    ) -> list[dict]:
        """Get recent events from MISP."""
        try:
            result = await self._request(
                "POST",
                "/events/restSearch",
                json={
                    "last": f"{days}d",
                    "limit": limit,
                },
            )
            return result.get("response", [])
        except Exception as e:
            self.logger.error("misp_events_failed", error=str(e))
            return []


class AbuseIPDBProvider(CachedProvider):
    """
    AbuseIPDB integration for IP reputation.
    
    Provides:
    - IP abuse reports
    - Confidence scores
    - Category classification
    - Historical data
    """
    
    BASE_URL = "https://api.abuseipdb.com/api/v2"
    
    def __init__(
        self,
        api_key: str,
        cache_ttl: int = 3600,
        timeout: float = 30.0,
    ):
        super().__init__(cache_ttl=cache_ttl)
        self.api_key = api_key
        self.timeout = timeout
        self._client = None
        self.logger = logger.bind(provider="abuseipdb")
    
    @property
    def name(self) -> str:
        return "abuseipdb"
    
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Key": self.api_key,
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
        return self._client
    
    async def enrich_ip(self, ip: str) -> dict:
        """Check IP reputation."""
        cache_key = self._cache_key("ip", ip)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            client = self._get_client()
            response = await client.get(
                "/check",
                params={
                    "ipAddress": ip,
                    "maxAgeInDays": 90,
                    "verbose": True,
                },
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            
            result = {
                "ip": ip,
                "abuse_confidence_score": data.get("abuseConfidenceScore", 0),
                "country": data.get("countryCode"),
                "isp": data.get("isp"),
                "domain": data.get("domain"),
                "total_reports": data.get("totalReports", 0),
                "num_distinct_users": data.get("numDistinctUsers", 0),
                "last_reported": data.get("lastReportedAt"),
                "is_whitelisted": data.get("isWhitelisted", False),
                "categories": data.get("reports", [{}])[0].get("categories", []) if data.get("reports") else [],
                "malicious": data.get("abuseConfidenceScore", 0) > 50,
                "suspicious": data.get("abuseConfidenceScore", 0) > 25,
            }
            
            await self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error("abuseipdb_check_failed", ip=ip, error=str(e))
            return {"ip": ip, "error": str(e)}
    
    async def enrich_domain(self, domain: str) -> dict:
        """AbuseIPDB doesn't support domain lookups."""
        return {"domain": domain, "error": "Domain lookups not supported"}
    
    async def enrich_hash(self, hash_value: str, hash_type: str) -> dict:
        """AbuseIPDB doesn't support hash lookups."""
        return {"hash": hash_value, "error": "Hash lookups not supported"}
    
    async def search_ioc(self, ioc: str) -> list[dict]:
        """Search is limited to IP for AbuseIPDB."""
        result = await self.enrich_ip(ioc)
        return [result] if "error" not in result else []


class STIXTAXIIProvider(CachedProvider):
    """
    STIX/TAXII feed integration.
    
    Supports:
    - TAXII 2.x servers
    - STIX 2.x objects
    - Collection enumeration
    - Indicator retrieval
    """
    
    def __init__(
        self,
        server_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_root: Optional[str] = None,
        collection_id: Optional[str] = None,
        cache_ttl: int = 1800,
        timeout: float = 60.0,
    ):
        super().__init__(cache_ttl=cache_ttl)
        self.server_url = server_url.rstrip("/")
        self.username = username
        self.password = password
        self.api_root = api_root
        self.collection_id = collection_id
        self.timeout = timeout
        self._client = None
        self.logger = logger.bind(provider="stix_taxii")
    
    @property
    def name(self) -> str:
        return "stix_taxii"
    
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            auth = None
            if self.username and self.password:
                auth = (self.username, self.password)
            
            self._client = httpx.AsyncClient(
                headers={
                    "Accept": "application/taxii+json;version=2.1",
                    "Content-Type": "application/taxii+json;version=2.1",
                },
                auth=auth,
                timeout=self.timeout,
            )
        return self._client
    
    async def get_collections(self) -> list[dict]:
        """Get available TAXII collections."""
        try:
            client = self._get_client()
            api_root = self.api_root or self.server_url
            response = await client.get(f"{api_root}/collections/")
            response.raise_for_status()
            return response.json().get("collections", [])
        except Exception as e:
            self.logger.error("taxii_collections_failed", error=str(e))
            return []
    
    async def get_objects(
        self,
        collection_id: Optional[str] = None,
        object_type: Optional[str] = None,
        added_after: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get STIX objects from collection."""
        cid = collection_id or self.collection_id
        if not cid:
            return []
        
        try:
            client = self._get_client()
            api_root = self.api_root or self.server_url
            
            params = {"limit": limit}
            if object_type:
                params["type"] = object_type
            if added_after:
                params["added_after"] = added_after.isoformat() + "Z"
            
            response = await client.get(
                f"{api_root}/collections/{cid}/objects/",
                params=params,
            )
            response.raise_for_status()
            return response.json().get("objects", [])
            
        except Exception as e:
            self.logger.error("taxii_objects_failed", error=str(e))
            return []
    
    async def enrich_ip(self, ip: str) -> dict:
        """Search STIX objects for IP."""
        return await self._search_indicator(f"ipv4-addr:value = '{ip}'")
    
    async def enrich_domain(self, domain: str) -> dict:
        """Search STIX objects for domain."""
        return await self._search_indicator(f"domain-name:value = '{domain}'")
    
    async def enrich_hash(self, hash_value: str, hash_type: str) -> dict:
        """Search STIX objects for hash."""
        hash_type_mapping = {
            "md5": "MD5",
            "sha1": "SHA-1",
            "sha256": "SHA-256",
        }
        stix_hash = hash_type_mapping.get(hash_type.lower(), hash_type.upper())
        return await self._search_indicator(f"file:hashes.'{stix_hash}' = '{hash_value}'")
    
    async def _search_indicator(self, pattern: str) -> dict:
        """Search for STIX indicator pattern."""
        cache_key = self._cache_key("indicator", pattern)
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            objects = await self.get_objects(object_type="indicator")
            
            matching = [
                obj for obj in objects
                if pattern in obj.get("pattern", "")
            ]
            
            result = {
                "pattern": pattern,
                "matches": len(matching),
                "indicators": [
                    {
                        "id": obj.get("id"),
                        "name": obj.get("name"),
                        "pattern": obj.get("pattern"),
                        "valid_from": obj.get("valid_from"),
                        "labels": obj.get("labels", []),
                    }
                    for obj in matching[:20]
                ],
                "malicious": len(matching) > 0,
            }
            
            await self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            self.logger.error("stix_search_failed", pattern=pattern, error=str(e))
            return {"pattern": pattern, "error": str(e)}
    
    async def search_ioc(self, ioc: str) -> list[dict]:
        """Generic IOC search."""
        results = []
        
        # Try different pattern types
        for pattern in [
            f"ipv4-addr:value = '{ioc}'",
            f"domain-name:value = '{ioc}'",
            f"file:hashes.'SHA-256' = '{ioc}'",
            f"file:hashes.'MD5' = '{ioc}'",
        ]:
            result = await self._search_indicator(pattern)
            if result.get("matches", 0) > 0:
                results.extend(result.get("indicators", []))
        
        return results


class ThreatFeedAggregator:
    """
    Aggregates threat intelligence from multiple providers.
    
    Features:
    - Parallel queries
    - Result deduplication
    - Confidence scoring
    - Source attribution
    """
    
    def __init__(self, providers: list[ThreatIntelProvider]):
        self.providers = providers
        self.logger = logger.bind(component="aggregator")
    
    async def enrich_ioc(
        self,
        ioc_type: str,
        ioc_value: str,
    ) -> dict:
        """
        Enrich IOC using all providers.
        
        Returns aggregated results with confidence scores.
        """
        tasks = []
        
        for provider in self.providers:
            if ioc_type in ["ipv4", "ipv6"]:
                tasks.append(self._safe_call(provider.enrich_ip, ioc_value))
            elif ioc_type == "domain":
                tasks.append(self._safe_call(provider.enrich_domain, ioc_value))
            elif ioc_type in ["md5", "sha1", "sha256"]:
                tasks.append(self._safe_call(provider.enrich_hash, ioc_value, ioc_type))
            else:
                tasks.append(self._safe_call(provider.search_ioc, ioc_value))
        
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        aggregated = {
            "ioc_type": ioc_type,
            "ioc_value": ioc_value,
            "sources": {},
            "is_malicious": False,
            "is_suspicious": False,
            "confidence": 0.0,
            "threat_actors": [],
            "campaigns": [],
        }
        
        malicious_count = 0
        suspicious_count = 0
        total_sources = 0
        
        for provider, result in zip(self.providers, results):
            if result and "error" not in result:
                aggregated["sources"][provider.name] = result
                total_sources += 1
                
                if result.get("malicious"):
                    malicious_count += 1
                if result.get("suspicious"):
                    suspicious_count += 1
                
                # Extract threat actors
                if "threat_actors" in result:
                    aggregated["threat_actors"].extend(result["threat_actors"])
                
                # Extract campaigns
                if "campaigns" in result:
                    aggregated["campaigns"].extend(result["campaigns"])
        
        # Calculate confidence
        if total_sources > 0:
            aggregated["is_malicious"] = malicious_count > 0
            aggregated["is_suspicious"] = suspicious_count > 0
            aggregated["confidence"] = (malicious_count + 0.5 * suspicious_count) / total_sources
        
        return aggregated
    
    async def _safe_call(self, func, *args) -> Optional[dict]:
        """Safely call provider function."""
        try:
            return await func(*args)
        except Exception as e:
            self.logger.warning("provider_call_failed", error=str(e))
            return None
