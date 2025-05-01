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
from rich.console import Console
from chatbot import VideoChatbot
from db_manager import DBManager

console = Console()

async def list_available_videos():
    """List all available videos in the database"""
    db_manager = DBManager()
    await db_manager.setup_database()
    
    async with db_manager.pool.acquire() as conn:
        results = await conn.fetch('''
            SELECT DISTINCT video_name FROM video_documents
            ORDER BY video_name
        ''')
        
        return [r['video_name'] for r in results]

async def main_async():
    """Main async function to run the chatbot"""
    parser = argparse.ArgumentParser(description="Video Chat Interface")
    parser.add_argument("--video", "-v", help="Name of the video to chat about (if not specified, will list available videos)")
    
    args = parser.parse_args()
    
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
    
    # Initialize and start the chatbot
    chatbot = VideoChatbot(video_name=video_name)
    await chatbot.setup()
    await chatbot.start_cli()

def main():
    """Entry point function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 