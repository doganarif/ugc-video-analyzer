import os
import json
from typing import List, Dict, Any, Optional
import asyncio
import asyncpg
from pydantic import BaseModel
import numpy as np
from models import VideoAnalysis

class VideoDocument(BaseModel):
    """Represents a document extracted from video analysis that will be stored in vector DB"""
    id: Optional[int] = None
    video_name: str 
    segment_timeframe: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None

class DBManager:
    def __init__(self, connection_string=None):
        """Initialize database manager with a connection string"""
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        self.pool = None
    
    async def init_pool(self):
        """Initialize connection pool with vector type support"""
        if not self.pool:
            # Define a function to convert Python lists to PostgreSQL vector format
            def encode_vector(value):
                return f"[{','.join(map(str, value))}]"
                
            # Create the connection pool with the vector type handler
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                init=lambda conn: conn.set_type_codec(
                    'vector',
                    encoder=encode_vector,
                    decoder=lambda value: value,
                    format='text'
                )
            )
            
    async def setup_database(self):
        """Setup database tables and extensions for vector search"""
        await self.init_pool()
        
        async with self.pool.acquire() as conn:
            # Create the pgvector extension if it doesn't exist
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            
            # Create video_documents table with vector support
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS video_documents (
                    id SERIAL PRIMARY KEY,
                    video_name TEXT NOT NULL,
                    segment_timeframe TEXT,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    embedding vector(1536)
                );
            ''')
            
            # Create index for faster vector search
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS video_documents_embedding_idx 
                ON video_documents 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            ''')
    
    async def store_document(self, document: VideoDocument, embedding: List[float]):
        """Store a document with its embedding in the database"""
        await self.init_pool()
        
        async with self.pool.acquire() as conn:
            document.embedding = embedding
            
            result = await conn.fetchval('''
                INSERT INTO video_documents(video_name, segment_timeframe, content, metadata, embedding)
                VALUES($1, $2, $3, $4, $5)
                RETURNING id
            ''', document.video_name, document.segment_timeframe, 
                document.content, json.dumps(document.metadata), embedding)
            
            return result
    
    async def store_analysis_document(self, analysis: VideoAnalysis, embedding: List[float]):
        """Store a document created from VideoAnalysis"""
        # Combine relevant fields into content
        content = f"{analysis.text_analysis}\n"
        if analysis.transcription:
            content += f"Transcription: {analysis.transcription}\n"
        
        # Create metadata from relevant fields
        metadata = {
            "analysis_type": analysis.analysis_type.value,
            "key_messages": analysis.key_messages,
            "people": [p.model_dump() for p in analysis.people],
            "content_details": analysis.content_details.model_dump(),
        }
        
        # Create document
        doc = VideoDocument(
            video_name=analysis.video_name,
            segment_timeframe=analysis.segment_timeframe,
            content=content,
            metadata=metadata
        )
        
        # Store document in DB
        return await self.store_document(doc, embedding)
    
    async def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents based on vector similarity"""
        await self.init_pool()
        
        async with self.pool.acquire() as conn:
            results = await conn.fetch('''
                SELECT id, video_name, segment_timeframe, content, metadata,
                       1 - (embedding <=> $1) as similarity
                FROM video_documents
                ORDER BY embedding <=> $1
                LIMIT $2
            ''', query_embedding, limit)
            
            return [dict(r) for r in results]
    
    async def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        await self.init_pool()
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow('''
                SELECT id, video_name, segment_timeframe, content, metadata
                FROM video_documents
                WHERE id = $1
            ''', doc_id)
            
            return dict(result) if result else None 