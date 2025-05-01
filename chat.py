#!/usr/bin/env python3
"""
Video Analysis Chatbot CLI
--------------------------
This script allows users to start a chatbot session 
to interact with previously analyzed videos.
"""

import os
import asyncio
import argparse
import logging
from rich.console import Console
from chatbot import VideoChatbot
from db_manager import DBManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

console = Console()

async def list_available_videos():
    """
    List all available videos in the database
    
    Returns:
        list: List of video names
    """
    try:
        db_manager = DBManager()
        await db_manager.setup_database()
        
        async with db_manager.pool.acquire() as conn:
            results = await conn.fetch('''
                SELECT DISTINCT video_name FROM video_documents
                ORDER BY video_name
            ''')
            
            return [r['video_name'] for r in results]
    except Exception as e:
        logger.error(f"Error listing videos: {e}")
        console.print(f"[bold red]Error listing videos: {e}[/bold red]")
        return []

async def main_async():
    """Main async function to run the chatbot"""
    parser = argparse.ArgumentParser(description="Video Chat Interface")
    parser.add_argument("--video", "-v", help="Name of the video to chat about (if not specified, will list available videos)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    if not args.video:
        # List available videos if none specified
        console.print("[bold yellow]No video specified. Listing available videos...[/bold yellow]")
        videos = await list_available_videos()
        
        if not videos:
            console.print("[bold red]No videos found in the database. Please run main.py to process a video first.[/bold red]")
            return
        
        console.print("[bold green]Available videos:[/bold green]")
        for i, video in enumerate(videos, 1):
            console.print(f"  {i}. {video}")
        
        # Let user select a video
        choice = -1
        while choice < 1 or choice > len(videos):
            try:
                choice = int(input(f"\nEnter video number (1-{len(videos)}): "))
            except ValueError:
                console.print("[bold red]Please enter a valid number.[/bold red]")
        
        video_name = videos[choice-1]
    else:
        video_name = args.video
    
    console.print(f"[bold green]Starting chat for video: {video_name}[/bold green]")
    logger.info(f"Starting chatbot for video: {video_name}")
    
    # Initialize and start the chatbot
    try:
        chatbot = VideoChatbot(video_name=video_name, debug_mode=args.debug)
        await chatbot.setup()
        await chatbot.start_cli()
    except Exception as e:
        logger.error(f"Error running chatbot: {e}")
        console.print(f"[bold red]Error running chatbot: {e}[/bold red]")

def main():
    """Entry point function"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Chatbot terminated by user.[/bold yellow]")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        console.print(f"\n[bold red]Unhandled error: {e}[/bold red]")

if __name__ == "__main__":
    main() 