"""Knowledge Graph extraction and management for wasmCloud RAG bot."""

import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from openai import OpenAI
from sqlalchemy.orm import Session
from sqlalchemy import text, and_, or_
import numpy as np
from dotenv import load_dotenv

from .models import Chunk, Entity, Relation, Triplet
from .embeddings import generate_query_embedding

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class KnowledgeGraphExtractor:
    """Extracts knowledge graph triplets from document chunks using GPT-4."""
    
    def __init__(self):
        self.extraction_model = "gpt-4-1106-preview"
        self.embedding_cache = {}
        
        # wasmCloud-specific entity types and relations
        self.entity_types = [
            "concept", "technology", "component", "protocol", "platform",
            "tool", "language", "framework", "service", "capability",
            "actor", "provider", "interface", "operation", "configuration"
        ]
        
        self.relation_types = [
            "functional", "hierarchical", "dependency", "implements",
            "provides", "requires", "uses", "configures", "manages",
            "deploys", "scales", "monitors", "connects", "supports"
        ]
    
    def extract_triplets_from_chunk(self, chunk: Chunk, db: Session) -> List[Dict[str, Any]]:
        """Extract knowledge triplets from a single chunk using GPT-4."""
        
        try:
            # Create extraction prompt
            prompt = self._create_extraction_prompt(chunk.content)
            
            # Call GPT-4 for extraction
            response = client.chat.completions.create(
                model=self.extraction_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000
            )
            
            # Parse response
            raw_triplets = self._parse_extraction_response(response.choices[0].message.content)
            
            # Process and store triplets
            processed_triplets = []
            for triplet_data in raw_triplets:
                triplet = self._process_and_store_triplet(triplet_data, chunk, db)
                if triplet:
                    processed_triplets.append(triplet)
            
            logger.info(f"Extracted {len(processed_triplets)} triplets from chunk {chunk.id}")
            return processed_triplets
            
        except Exception as e:
            logger.error(f"Error extracting triplets from chunk {chunk.id}: {e}")
            return []
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for triplet extraction."""
        return """You are an expert at extracting structured knowledge from wasmCloud documentation. 

Your task is to identify relationships between entities in the text and express them as triplets in the format:
(subject, predicate, object)

Focus on wasmCloud-specific concepts including:
- Actors, capabilities, providers, interfaces
- WASM modules, OCI registries, NATS messaging
- Deployment, scaling, configuration concepts
- Platform integrations (Kubernetes, cloud providers)
- Development tools and workflows

Entity types to recognize:
- concept, technology, component, protocol, platform, tool, language, framework, service, capability, actor, provider, interface, operation, configuration

Relation types to use:
- implements, provides, requires, uses, configures, manages, deploys, scales, monitors, connects, supports, runs_on, part_of, enables, contains

Return ONLY valid JSON with an array of triplets. Each triplet should have:
- subject: entity name
- subject_type: entity type
- predicate: relation name
- predicate_type: relation type  
- object: entity name
- object_type: entity type
- confidence: float 0.0-1.0
- context: the sentence containing this relationship
- extracted_text: the specific text span that led to this triplet

Example format:
[
  {
    "subject": "wasmCloud Actor",
    "subject_type": "component",
    "predicate": "implements",
    "predicate_type": "functional",
    "object": "WebAssembly Interface",
    "object_type": "interface",
    "confidence": 0.9,
    "context": "wasmCloud actors implement WebAssembly interfaces to provide capabilities.",
    "extracted_text": "wasmCloud actors implement WebAssembly interfaces"
  }
]"""
    
    def _create_extraction_prompt(self, content: str) -> str:
        """Create the extraction prompt for a specific chunk."""
        # Limit content length to avoid token limits
        max_chars = 3000
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        return f"""Extract knowledge triplets from this wasmCloud documentation text:

{content}

Focus on relationships between technical concepts, components, and processes. 
Return structured JSON with triplets as specified in the system prompt."""
    
    def _parse_extraction_response(self, response_content: str) -> List[Dict[str, Any]]:
        """Parse GPT-4 response to extract triplet data."""
        try:
            # Clean the response - remove markdown code blocks if present
            clean_content = response_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            clean_content = clean_content.strip()
            
            # Parse JSON
            triplets_data = json.loads(clean_content)
            
            # Validate structure
            if not isinstance(triplets_data, list):
                logger.warning("Response is not a list, wrapping in list")
                triplets_data = [triplets_data] if triplets_data else []
            
            # Filter and validate triplets
            valid_triplets = []
            for triplet in triplets_data:
                if self._validate_triplet_data(triplet):
                    valid_triplets.append(triplet)
                else:
                    logger.warning(f"Invalid triplet data: {triplet}")
            
            return valid_triplets
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response content: {response_content}")
            return []
        except Exception as e:
            logger.error(f"Error parsing extraction response: {e}")
            return []
    
    def _validate_triplet_data(self, triplet: Dict[str, Any]) -> bool:
        """Validate that triplet data has required fields."""
        required_fields = ["subject", "predicate", "object", "confidence"]
        return all(field in triplet for field in required_fields)
    
    def _process_and_store_triplet(self, triplet_data: Dict[str, Any], chunk: Chunk, db: Session) -> Optional[Dict[str, Any]]:
        """Process and store a single triplet in the database."""
        try:
            # Get or create entities
            subject_entity = self._get_or_create_entity(
                name=triplet_data["subject"],
                entity_type=triplet_data.get("subject_type", "concept"),
                description=triplet_data.get("context", ""),
                db=db
            )
            
            object_entity = self._get_or_create_entity(
                name=triplet_data["object"],
                entity_type=triplet_data.get("object_type", "concept"),
                description=triplet_data.get("context", ""),
                db=db
            )
            
            # Get or create relation
            relation = self._get_or_create_relation(
                name=triplet_data["predicate"],
                relation_type=triplet_data.get("predicate_type", "functional"),
                db=db
            )
            
            # Check if triplet already exists
            existing_triplet = db.query(Triplet).filter(
                and_(
                    Triplet.subject_id == subject_entity.id,
                    Triplet.relation_id == relation.id,
                    Triplet.object_id == object_entity.id,
                    Triplet.source_chunk_id == chunk.id
                )
            ).first()
            
            if existing_triplet:
                logger.debug(f"Triplet already exists: {triplet_data['subject']} -> {triplet_data['predicate']} -> {triplet_data['object']}")
                return None
            
            # Create new triplet
            new_triplet = Triplet(
                subject_id=subject_entity.id,
                relation_id=relation.id,
                object_id=object_entity.id,
                source_chunk_id=chunk.id,
                confidence_score=float(triplet_data["confidence"]),
                context_sentence=triplet_data.get("context", ""),
                extracted_text=triplet_data.get("extracted_text", "")
            )
            
            db.add(new_triplet)
            db.commit()
            
            return {
                "id": new_triplet.id,
                "subject": subject_entity.name,
                "predicate": relation.name,
                "object": object_entity.name,
                "confidence": new_triplet.confidence_score
            }
            
        except Exception as e:
            logger.error(f"Error processing triplet: {e}")
            db.rollback()
            return None
    
    def _get_or_create_entity(self, name: str, entity_type: str, description: str, db: Session) -> Entity:
        """Get existing entity or create new one."""
        # Normalize entity name
        canonical_form = self._normalize_entity_name(name)
        
        # Try to find existing entity
        entity = db.query(Entity).filter(
            and_(
                Entity.canonical_form == canonical_form,
                Entity.entity_type == entity_type
            )
        ).first()
        
        if entity:
            # Update frequency
            entity.frequency += 1
            db.commit()
            return entity
        
        # Create new entity
        entity_embedding = None
        try:
            entity_embedding = generate_query_embedding(f"{name} {entity_type}")
        except Exception as e:
            logger.warning(f"Failed to generate embedding for entity {name}: {e}")
        
        new_entity = Entity(
            name=name,
            entity_type=entity_type,
            description=description[:500] if description else None,  # Limit description length
            canonical_form=canonical_form,
            embedding=entity_embedding,
            frequency=1
        )
        
        db.add(new_entity)
        db.commit()
        return new_entity
    
    def _get_or_create_relation(self, name: str, relation_type: str, db: Session) -> Relation:
        """Get existing relation or create new one."""
        # Normalize relation name
        normalized_name = name.lower().strip()
        
        # Try to find existing relation
        relation = db.query(Relation).filter(Relation.name == normalized_name).first()
        
        if relation:
            # Update frequency
            relation.frequency += 1
            db.commit()
            return relation
        
        # Create new relation
        new_relation = Relation(
            name=normalized_name,
            relation_type=relation_type,
            frequency=1
        )
        
        db.add(new_relation)
        db.commit()
        return new_relation
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication."""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', name.strip().lower())
        
        # Handle common variations
        replacements = {
            'wasmcloud': 'wasmcloud',
            'wasm cloud': 'wasmcloud',
            'webassembly': 'webassembly',
            'web assembly': 'webassembly',
            'kubernetes': 'kubernetes',
            'k8s': 'kubernetes',
            'docker': 'docker',
            'oci': 'oci'
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized


class KnowledgeGraphRetriever:
    """Retrieves relevant knowledge graph information for queries."""
    
    def __init__(self):
        self.embedding_cache = {}
    
    def retrieve_kg_context(self, query: str, db: Session, max_triplets: int = 20) -> Dict[str, Any]:
        """Retrieve relevant knowledge graph context for a query."""
        
        try:
            # Find relevant entities
            relevant_entities = self._find_relevant_entities(query, db)
            
            # Find relevant triplets
            relevant_triplets = self._find_relevant_triplets(query, relevant_entities, db, max_triplets)
            
            # Build knowledge graph context
            kg_context = self._build_kg_context(relevant_triplets, db)
            
            return {
                "triplets": relevant_triplets,
                "context": kg_context,
                "entity_count": len(relevant_entities),
                "triplet_count": len(relevant_triplets)
            }
            
        except Exception as e:
            logger.error(f"Error retrieving KG context: {e}")
            return {"triplets": [], "context": "", "entity_count": 0, "triplet_count": 0}
    
    def _find_relevant_entities(self, query: str, db: Session, top_k: int = 10) -> List[Entity]:
        """Find entities relevant to the query."""
        try:
            # Generate query embedding
            query_embedding = generate_query_embedding(query)
            
            # Use vector similarity if embeddings are available and pgvector is available
            if query_embedding:
                try:
                    # Check if we have pgvector support
                    from .models import PGVECTOR_AVAILABLE
                    
                    if PGVECTOR_AVAILABLE:
                        # Vector similarity search with pgvector
                        sql_query = text("""
                            SELECT e.*, 
                                   (e.embedding <=> :query_embedding) as distance
                            FROM entities e
                            WHERE e.embedding IS NOT NULL
                            ORDER BY distance ASC
                            LIMIT :limit
                        """)
                        
                        result = db.execute(sql_query, {
                            "query_embedding": query_embedding,
                            "limit": top_k
                        })
                        
                        entities = []
                        for row in result:
                            entity = db.query(Entity).get(row.id)
                            if entity:
                                entities.append(entity)
                        
                        if entities:
                            return entities
                            
                except Exception as e:
                    logger.warning(f"Vector similarity search failed, falling back to text search: {e}")
            
            # Fallback to text-based search
            query_words = query.lower().split()
            entities = db.query(Entity).filter(
                or_(*[Entity.name.ilike(f"%{word}%") for word in query_words])
            ).order_by(Entity.frequency.desc()).limit(top_k).all()
            
            return entities
            
        except Exception as e:
            logger.error(f"Error finding relevant entities: {e}")
            return []
    
    def _find_relevant_triplets(self, query: str, entities: List[Entity], db: Session, max_triplets: int) -> List[Dict[str, Any]]:
        """Find triplets involving relevant entities."""
        
        if not entities:
            return []
        
        entity_ids = [e.id for e in entities]
        
        # Find triplets where entities are subjects or objects
        triplets = db.query(Triplet).filter(
            or_(
                Triplet.subject_id.in_(entity_ids),
                Triplet.object_id.in_(entity_ids)
            )
        ).order_by(Triplet.confidence_score.desc()).limit(max_triplets).all()
        
        # Convert to dicts with entity names
        triplet_dicts = []
        for triplet in triplets:
            triplet_dicts.append({
                "id": triplet.id,
                "subject": triplet.subject.name,
                "subject_type": triplet.subject.entity_type,
                "predicate": triplet.relation.name,
                "predicate_type": triplet.relation.relation_type,
                "object": triplet.object.name,
                "object_type": triplet.object.entity_type,
                "confidence": triplet.confidence_score,
                "context": triplet.context_sentence,
                "source_chunk_id": triplet.source_chunk_id
            })
        
        return triplet_dicts
    
    def _build_kg_context(self, triplets: List[Dict[str, Any]], db: Session) -> str:
        """Build a readable knowledge graph context from triplets."""
        
        if not triplets:
            return ""
        
        context_parts = ["**Knowledge Graph Context:**\n"]
        
        # Group triplets by relation type
        relation_groups = {}
        for triplet in triplets:
            rel_type = triplet["predicate_type"]
            if rel_type not in relation_groups:
                relation_groups[rel_type] = []
            relation_groups[rel_type].append(triplet)
        
        # Build context by relation type
        for rel_type, group_triplets in relation_groups.items():
            context_parts.append(f"\n**{rel_type.title()} Relationships:**")
            
            for triplet in group_triplets[:5]:  # Limit per group
                context_parts.append(
                    f"- {triplet['subject']} {triplet['predicate']} {triplet['object']} "
                    f"(confidence: {triplet['confidence']:.2f})"
                )
        
        return "\n".join(context_parts)


# Utility functions for batch processing
def extract_triplets_from_all_chunks(db: Session, batch_size: int = 50) -> Dict[str, int]:
    """Extract triplets from all chunks that don't have triplets yet."""
    
    extractor = KnowledgeGraphExtractor()
    
    # Find chunks without triplets
    chunks_without_triplets = db.query(Chunk).filter(
        ~Chunk.triplets.any()
    ).all()
    
    total_chunks = len(chunks_without_triplets)
    total_triplets = 0
    
    logger.info(f"Starting triplet extraction for {total_chunks} chunks")
    
    for i, chunk in enumerate(chunks_without_triplets):
        try:
            triplets = extractor.extract_triplets_from_chunk(chunk, db)
            total_triplets += len(triplets)
            
            if (i + 1) % batch_size == 0:
                logger.info(f"Processed {i + 1}/{total_chunks} chunks, extracted {total_triplets} triplets so far")
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.id}: {e}")
            continue
    
    logger.info(f"Completed triplet extraction: {total_triplets} triplets from {total_chunks} chunks")
    
    return {
        "chunks_processed": total_chunks,
        "triplets_extracted": total_triplets
    }


def get_knowledge_graph_stats(db: Session) -> Dict[str, Any]:
    """Get statistics about the knowledge graph."""
    
    try:
        entity_count = db.query(Entity).count()
        relation_count = db.query(Relation).count()
        triplet_count = db.query(Triplet).count()
        
        # Top entities by frequency
        top_entities = db.query(Entity).order_by(Entity.frequency.desc()).limit(10).all()
        
        # Top relations by frequency
        top_relations = db.query(Relation).order_by(Relation.frequency.desc()).limit(10).all()
        
        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "triplet_count": triplet_count,
            "top_entities": [{"name": e.name, "type": e.entity_type, "frequency": e.frequency} for e in top_entities],
            "top_relations": [{"name": r.name, "type": r.relation_type, "frequency": r.frequency} for r in top_relations]
        }
        
    except Exception as e:
        logger.error(f"Error getting KG stats: {e}")
        return {} 