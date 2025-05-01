import os
from config import PromptType, DEFAULT_TEMPERATURE
from analyzer import execute_video_processing, create_comprehensive_analysis, analyze_video_with_multiple_perspectives

def main():
    """
    Main function demonstrating usage of the UGC video and ad analysis functionality
    """
    video_file = 'sample.mp4'  # Replace with the path to your ad/UGC video file
    output_analysis_dir = 'analysis_output'
    os.makedirs(output_analysis_dir, exist_ok=True)

    # Example 1: Process the entire video with general ad analysis
    print(f"\n--- Processing the entire ad with general analysis: {video_file} ---")
    general_analysis = execute_video_processing(
        shot_path=video_file,
        analysis_type=PromptType.GENERAL,
        temperature=DEFAULT_TEMPERATURE,
        frames_per_second=1,  # Lower frame rate for efficiency
        analysis_dir=output_analysis_dir,
        save_frames=False,
        audio_transcription=True
    )
    print(f"\nGeneral Analysis Summary:\n{general_analysis.model_dump_json(indent=2)}")

    # Example 2: Process the video with audience targeting analysis
    print(f"\n--- Processing the video with audience targeting analysis: {video_file} ---")
    audience_analysis = execute_video_processing(
        shot_path=video_file,
        analysis_type=PromptType.AUDIENCE,
        temperature=DEFAULT_TEMPERATURE,
        frames_per_second=0.5,  # Even lower frame rate for audience analysis
        analysis_dir=output_analysis_dir,
        save_frames=False,
        audio_transcription=True
    )
    print(f"\nAudience Analysis Summary:\n{audience_analysis.model_dump_json(indent=2)}")

    # Example 3: Process the video in segments with comprehensive analysis
    print(f"\n--- Processing video segments with comprehensive analysis: {video_file} ---")
    shot_interval = 5  # Analyze every 5 seconds - shorter for ads
    comprehensive_analysis = create_comprehensive_analysis(
        video_path=video_file,
        output_dir=output_analysis_dir,
        shot_interval=shot_interval,
        max_duration=60,  # Example max duration of 60 seconds
        save_frames=False,
        audio_transcription=True
    )

    # Print summary of comprehensive analysis
    print("\nComprehensive Analysis Summary:")
    for timeframe, analyses in comprehensive_analysis.items():
        print(f"  Segment {timeframe}:")
        for analysis_type, result in analyses.items():
            print(f"    - {analysis_type}: {result.content_details.setting}")

    print("\nUGC Video & Ad Analyzer usage examples:")
    print("""
    # Basic usage - analyze a single ad/UGC video with general analysis
    general_analysis = execute_video_processing(
        shot_path='your_ad.mp4',
        analysis_type=PromptType.GENERAL,
        audio_transcription=True
    )
    
    # Audience targeting analysis
    audience_analysis = execute_video_processing(
        shot_path='your_ad.mp4', 
        analysis_type=PromptType.AUDIENCE,
        audio_transcription=True
    )
    
    # Visual branding analysis
    visual_analysis = execute_video_processing(
        shot_path='your_ad.mp4',
        analysis_type=PromptType.VISUAL,
        audio_transcription=True
    )
    
    # Call-to-action and conversion analysis
    conversion_analysis = execute_video_processing(
        shot_path='your_ad.mp4',
        analysis_type=PromptType.CONVERSION,
        audio_transcription=True
    )
    
    # For a multi-perspective breakdown of a social media ad
    analysis = analyze_video_with_multiple_perspectives(
        video_path='your_ad.mp4',
        shot_interval=5,  # 5-second segments for shorter content
        analysis_types=[PromptType.GENERAL, PromptType.AUDIENCE, PromptType.ENGAGEMENT]
    )
    
    # For a complete marketing effectiveness analysis with all perspectives
    full_analysis = create_comprehensive_analysis(
        video_path='your_ad.mp4',
        shot_interval=10  # 10-second segments
    )
    """)

if __name__ == '__main__':
    main() 