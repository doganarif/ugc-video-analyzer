import os
import asyncio
from typing import List, Dict, Any
import argparse
from datetime import datetime

from config import PromptType, DEFAULT_TEMPERATURE
from analyzer import execute_video_processing, create_comprehensive_analysis, analyze_video_with_multiple_perspectives
from db_manager import DBManager
from embeddings_manager import EmbeddingsManager
from models import VideoAnalysis
from chatbot import VideoChatbot

async def process_segment(video_path: str, timeframe: str, analysis_types: List[PromptType], output_dir: str, db_manager: DBManager, embeddings_manager: EmbeddingsManager, save_frames: bool = False):
    """Process a single video segment asynchronously"""
    results = {}
    
    for analysis_type in analysis_types:
        print(f"Processing {timeframe} with {analysis_type.value} analysis...")
        
        # Execute video processing for this segment
        analysis = execute_video_processing(
            shot_path=video_path,
            analysis_type=analysis_type,
            temperature=DEFAULT_TEMPERATURE,
            frames_per_second=1,
            analysis_dir=output_dir,
            segment_timeframe=timeframe,
            save_frames=save_frames,
            audio_transcription=True
        )
        
        # Generate embedding for this analysis
        text_for_embedding = f"{analysis.text_analysis}\n"
        if analysis.transcription:
            text_for_embedding += f"Transcription: {analysis.transcription}"
        
        embedding = await embeddings_manager.generate_embedding_async(text_for_embedding)
        
        # Store in vector DB
        doc_id = await db_manager.store_analysis_document(analysis, embedding)
        print(f"Stored analysis in DB with ID: {doc_id}")
        
        results[analysis_type.value] = analysis
        
    return timeframe, results

async def async_comprehensive_analysis(video_path: str, output_dir: str, analysis_types: List[PromptType] = None, shot_interval: int = 10, max_duration: int = None, save_frames: bool = False):
    """Process a video with multiple analyses asynchronously"""
    if analysis_types is None:
        analysis_types = [PromptType.GENERAL, PromptType.AUDIENCE, PromptType.VISUAL]
    
    # Initialize DB and embeddings managers
    db_manager = DBManager()
    await db_manager.setup_database()
    embeddings_manager = EmbeddingsManager()
    
    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video duration and calculate segments
    import cv2
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    video.release()
    
    if max_duration and duration > max_duration:
        duration = max_duration
    
    # Calculate timeframes
    timeframes = []
    for start_time in range(0, int(duration), shot_interval):
        end_time = min(start_time + shot_interval, int(duration))
        timeframe = f"{start_time}s-{end_time}s"
        timeframes.append(timeframe)
    
    # Create tasks for parallel processing
    tasks = []
    for timeframe in timeframes:
        task = process_segment(
            video_path=video_path,
            timeframe=timeframe,
            analysis_types=analysis_types,
            output_dir=output_dir,
            db_manager=db_manager,
            embeddings_manager=embeddings_manager,
            save_frames=save_frames
        )
        tasks.append(task)
    
    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks)
    
    # Organize results by timeframe
    comprehensive_results = {}
    for timeframe, segment_results in results:
        comprehensive_results[timeframe] = segment_results
    
    # Create a final summary analysis
    full_video_analysis = execute_video_processing(
        shot_path=video_path,
        analysis_type=PromptType.GENERAL,
        temperature=DEFAULT_TEMPERATURE,
        frames_per_second=0.5,  # Lower FPS for full video
        analysis_dir=output_dir,
        save_frames=False,
        audio_transcription=True
    )
    
    # Generate embedding for full analysis
    text_for_embedding = f"{full_video_analysis.text_analysis}\n"
    if full_video_analysis.transcription:
        text_for_embedding += f"Transcription: {full_video_analysis.transcription}"
    
    embedding = await embeddings_manager.generate_embedding_async(text_for_embedding)
    
    # Store in vector DB
    doc_id = await db_manager.store_analysis_document(full_video_analysis, embedding)
    print(f"Stored full video analysis in DB with ID: {doc_id}")
    
    return comprehensive_results

async def main_async():
    """Async main function for video processing and chatbot"""
    parser = argparse.ArgumentParser(description="Video Analysis with Vector DB")
    parser.add_argument("--video", "-v", required=True, help="Path to the video file")
    parser.add_argument("--output", "-o", default="analysis_output", help="Output directory for analysis")
    parser.add_argument("--interval", "-i", type=int, default=10, help="Interval in seconds for segmenting the video")
    parser.add_argument("--chat", "-c", action="store_true", help="Start chatbot after analysis")
    parser.add_argument("--analysis-types", "-a", nargs="+", 
                        choices=[t.value for t in PromptType], 
                        default=["general", "audience", "visual"],
                        help="Types of analysis to perform")
    
    args = parser.parse_args()
    
    # Convert string analysis types to enum values
    analysis_types = [PromptType(t) for t in args.analysis_types]
    
    print(f"Starting analysis of {args.video} with {', '.join(args.analysis_types)} analysis types...")
    print(f"Segmenting video into {args.interval}-second chunks")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}/{os.path.basename(args.video)}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process video asynchronously
    results = await async_comprehensive_analysis(
        video_path=args.video,
        output_dir=output_dir,
        analysis_types=analysis_types,
        shot_interval=args.interval,
        save_frames=False
    )
    
    # Print summary
    print("\nAnalysis complete! Summary:")
    for timeframe, analyses in results.items():
        print(f"  Segment {timeframe}:")
        for analysis_type, result in analyses.items():
            print(f"    - {analysis_type}: {result.content_details.setting}")
    
    # Start chatbot if requested
    if args.chat:
        print("\nStarting chatbot for video interactions...")
        video_name = os.path.basename(args.video)
        chatbot = VideoChatbot(video_name=video_name)
        await chatbot.setup()
        await chatbot.start_cli()

def main():
    """Entry point function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 