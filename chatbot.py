import asyncio
import os
import argparse
import json
from openai import AzureOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from db_manager import DBManager
from embeddings_manager import EmbeddingsManager
from config import get_openai_api_key, get_chat_model, get_database_url

console = Console()

class VideoChatbot:
    """CLI Chatbot for interacting with analyzed video content"""

    def __init__(self, video_name: str, api_key=None, model=None):
        """Initialize the chatbot with video name and OpenAI credentials"""
        self.video_name = video_name
        self.api_key = api_key or get_openai_api_key()
        self.model = model or get_chat_model()

        # Get Azure OpenAI endpoint and API version from environment variables
        self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        self.azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Use AzureOpenAI client instead of OpenAI
        self.openai_client = AzureOpenAI(
            azure_deployment=self.azure_deployment,
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key
        )

        self.db_manager = DBManager(get_database_url())
        self.embeddings_manager = EmbeddingsManager(
            api_key=self.api_key,
            use_azure=True,
            azure_endpoint=self.azure_endpoint,
            azure_api_version=self.azure_api_version,
            azure_deployment="text-embedding-3-large"  # Make sure to use a valid embedding model
        )
        self.conversation_history = []
        self.debug_mode = True  # Set to True to see detailed debugging info

    async def setup(self):
        """Setup the database and connection pool"""
        await self.db_manager.setup_database()
        
    async def get_all_video_documents(self, limit=10):
        """Get all available video documents for the current video"""
        await self.db_manager.init_pool()
        
        async with self.db_manager.pool.acquire() as conn:
            results = await conn.fetch('''
                SELECT id, video_name, segment_timeframe, content, metadata
                FROM video_documents
                WHERE video_name = $1
                LIMIT $2
            ''', self.video_name, limit)
            
            return [dict(r) for r in results]
            
    async def search_by_keywords(self, keywords, limit=5):
        """Search for documents containing specific keywords"""
        await self.db_manager.init_pool()
        
        # Convert keywords to an array of search terms
        if isinstance(keywords, str):
            search_terms = [term.strip() for term in keywords.split() if len(term.strip()) > 3]
        elif isinstance(keywords, list):
            search_terms = keywords
        else:
            search_terms = []
            
        if not search_terms:
            return []
            
        # Use basic text search 
        search_queries = []
        for term in search_terms:
            if len(term) > 3:  # Only use terms with more than 3 characters
                search_queries.append(f"content ILIKE '%{term}%'")
                
        if not search_queries:
            return []
            
        query = " OR ".join(search_queries)
        
        async with self.db_manager.pool.acquire() as conn:
            results = await conn.fetch(f'''
                SELECT id, video_name, segment_timeframe, content, metadata
                FROM video_documents
                WHERE video_name = $1 AND ({query})
                LIMIT $2
            ''', self.video_name, limit)
            
            docs = [dict(r) for r in results]
            
            # Add a relevance score for each document based on keyword matches
            for doc in docs:
                content = doc.get('content', '').lower()
                score = 0
                for term in search_terms:
                    if term.lower() in content:
                        score += 1
                doc['similarity'] = min(0.8, score / len(search_terms))
                
            return docs

    async def get_relevant_context(self, query: str, max_results: int = 5) -> str:
        """Get relevant context from the vector database based on query using function calling"""
        try:
            # Use function calling to generate embedding and search
            console.print("[dim]Getting search terms from AI...[/dim]")
            
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that helps with video content retrieval. Extract the most important search terms from the user's query."},
                    {"role": "user", "content": f"Find relevant information about: {query}"}
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "search_video_content",
                            "description": "Search for relevant video content segments based on the query",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "search_query": {
                                        "type": "string",
                                        "description": "The search query to find relevant video content"
                                    },
                                    "search_terms": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                        "description": "Individual search terms extracted from the query"
                                    }
                                },
                                "required": ["search_query", "search_terms"]
                            }
                        }
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "search_video_content"}}
            )

            # Extract the search query from the function call
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            search_query = function_args["search_query"]
            search_terms = function_args.get("search_terms", [])
            
            console.print(f"[dim]Searching for: {search_query}[/dim]")
            if search_terms:
                console.print(f"[dim]Search terms: {', '.join(search_terms)}[/dim]")

            # STEP 1: Try vector search first
            console.print("[dim]Trying vector search...[/dim]")
            query_embedding = await self.embeddings_manager.generate_embedding_async(search_query)
            vector_results = await self.db_manager.search_similar(query_embedding, limit=max_results)
            
            # Check if we got good vector results (similarity > 0.3)
            good_vector_results = [r for r in vector_results if r.get('similarity', 0) > 0.3]
            
            if good_vector_results:
                console.print(f"[dim]Found {len(good_vector_results)} relevant results with vector search[/dim]")
                results = good_vector_results
            else:
                # STEP 2: If vector search fails, try keyword search
                console.print("[dim]Vector search returned weak results, trying keyword search...[/dim]")
                keyword_results = await self.search_by_keywords(search_terms, limit=max_results)
                
                if keyword_results:
                    console.print(f"[dim]Found {len(keyword_results)} results with keyword search[/dim]")
                    results = keyword_results
                else:
                    # STEP 3: If both fail, fall back to getting all documents
                    console.print("[dim]No keyword results, falling back to retrieving all documents...[/dim]")
                    results = await self.get_all_video_documents(limit=max_results)
                    
                    # Add a placeholder similarity score
                    for result in results:
                        if 'similarity' not in result:
                            result['similarity'] = 0.5

            if not results:
                return "No relevant information found for this query."

            # Build context from results
            context_parts = []
            for result in results:
                timeframe = result.get('segment_timeframe', 'unknown timeframe')
                similarity = result.get('similarity', 0)
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                
                # Extract useful information from metadata if available
                metadata_info = ""
                if metadata:
                    if 'key_messages' in metadata and metadata['key_messages']:
                        metadata_info += f"Key messages: {', '.join(metadata['key_messages'])}\n"
                    if 'people' in metadata and metadata['people']:
                        people_info = []
                        for person in metadata['people']:
                            if isinstance(person, dict) and 'role' in person and 'description' in person:
                                people_info.append(f"{person['role']} ({person['description']})")
                        if people_info:
                            metadata_info += f"People: {', '.join(people_info)}\n"
                    
                    # Extract content details if available
                    if 'content_details' in metadata and isinstance(metadata['content_details'], dict):
                        content_details = metadata['content_details']
                        if 'setting' in content_details:
                            metadata_info += f"Setting: {content_details['setting']}\n"
                        if 'platform' in content_details and content_details['platform']:
                            metadata_info += f"Platform: {content_details['platform']}\n"
                        if 'message_purpose' in content_details and content_details['message_purpose']:
                            metadata_info += f"Purpose: {content_details['message_purpose']}\n"

                # Use all results with any similarity
                segment_info = f"From segment {timeframe}"
                if 'similarity' in result:
                    segment_info += f" (relevance: {similarity:.2f})"
                segment_info += ":\n"
                
                context_parts.append(f"{segment_info}{content}\n{metadata_info}")

            if not context_parts:
                return "No relevant information found for this query, but here are some general details about the video."
                
            return "\n".join(context_parts)
            
        except Exception as e:
            console.print(f"[bold red]Error in get_relevant_context: {str(e)}[/bold red]")
            # Fallback to returning any information about the video
            try:
                results = await self.get_all_video_documents(limit=3)
                if results:
                    context_parts = []
                    for result in results:
                        timeframe = result.get('segment_timeframe', 'unknown timeframe')
                        content = result.get('content', '')
                        context_parts.append(f"From segment {timeframe}:\n{content}\n")
                    return "\n".join(context_parts)
                else:
                    return "Could not retrieve information about the video due to an error."
            except Exception as fallback_error:
                console.print(f"[bold red]Fallback error: {str(fallback_error)}[/bold red]")
                return "Could not retrieve information about the video due to an error."

    async def generate_response(self, user_query: str) -> str:
        """Generate a response to the user query using RAG"""
        # Get relevant context
        context = await self.get_relevant_context(user_query)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_query})

        # Construct system message with context
        system_message = f"""You are an AI assistant that helps users understand videos. 
You answer questions about the video titled '{self.video_name}'.
Use the following information retrieved from the video analysis to inform your answer:

{context}

If the information provided doesn't address the user's question directly, 
acknowledge this and use your general knowledge to give the best possible response,
but make it clear which parts are not directly from the video analysis."""

        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_message},
        ] + self.conversation_history

        # Call the Azure OpenAI API
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )

        # Get the assistant's response
        assistant_response = response.choices[0].message.content

        # Update conversation history
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

        return assistant_response

    async def start_cli(self):
        """Start the CLI interface for the chatbot"""
        console.print(Panel.fit(
            f"[bold green]Video Chatbot[/bold green]\n"
            f"Ask questions about the video: [italic]{self.video_name}[/italic]",
            title="Welcome"
        ))

        while True:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

            if user_input.lower() in ['exit', 'quit', 'bye']:
                console.print("[bold green]Goodbye![/bold green]")
                break

            with console.status("[bold green]Thinking...[/bold green]"):
                response = await self.generate_response(user_input)

            console.print("\n[bold green]AI Assistant[/bold green]")
            console.print(Markdown(response))
