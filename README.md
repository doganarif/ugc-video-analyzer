# Social Media & UGC Video Analyzer

A comprehensive tool for analyzing social media ads and user-generated content (UGC) videos using Azure OpenAI's GPT-4o, providing insights on marketing effectiveness, audience targeting, visual branding, and more.

## Features

- Frame extraction and processing from video files
- Audio transcription using Azure OpenAI Whisper
- Multiple analysis perspectives:
  - General ad analysis
  - Target audience analysis
  - Visual branding analysis
  - Marketing messaging analysis
  - Brand identity analysis
  - Engagement potential analysis
  - Technical production analysis
  - Conversion optimization analysis
- Segment-based analysis for longer ads
- Consolidated analysis across all segments
- Structured data output in JSON format

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on the provided `env_example` with your Azure OpenAI API credentials:
   ```
   cp env_example .env
   # Edit .env with your credentials
   ```

## Usage

### Basic Analysis

Analyze a single ad or UGC video with general analysis:

```python
from analyzer import execute_video_processing
from config import PromptType

general_analysis = execute_video_processing(
    shot_path='your_ad.mp4',
    analysis_type=PromptType.GENERAL,
    audio_transcription=True
)
```

### Specific Analysis Types

Perform audience targeting analysis:

```python
audience_analysis = execute_video_processing(
    shot_path='your_ad.mp4',
    analysis_type=PromptType.AUDIENCE,
    audio_transcription=True
)
```

Analyze conversion elements and call-to-action:

```python
conversion_analysis = execute_video_processing(
    shot_path='your_ad.mp4',
    analysis_type=PromptType.CONVERSION,
    audio_transcription=True
)
```

### Segmented Analysis

Break a video into segments and analyze each part:

```python
from analyzer import analyze_video_with_multiple_perspectives

analysis = analyze_video_with_multiple_perspectives(
    video_path='your_ad.mp4',
    shot_interval=5,  # 5-second segments for shorter content
    analysis_types=[PromptType.GENERAL, PromptType.AUDIENCE, PromptType.ENGAGEMENT]
)
```

### Comprehensive Analysis

Analyze a video from all marketing perspectives:

```python
from analyzer import create_comprehensive_analysis

full_analysis = create_comprehensive_analysis(
    video_path='your_ad.mp4',
    shot_interval=10  # 10-second segments
)
```

## Project Structure

- `config.py` - Configuration parameters, constants, and prompt templates
- `models.py` - Pydantic models for structured data
- `video_processor.py` - Video processing functions
- `audio_processor.py` - Audio transcription functions
- `analyzer.py` - Core analysis logic
- `main.py` - Main execution and examples
- `requirements.txt` - Project dependencies

## Example

Run the example script:

```
python main.py
```

This will analyze the sample video with different approaches and save the results in the `analysis_output` directory.

## Notes

- This tool requires Azure OpenAI services with access to GPT-4o and Whisper models
- Processing longer videos can be expensive in terms of API costs
- Default settings are optimized for short-form content (15-60 seconds)
- Frame extraction rate and segment intervals can be adjusted for different video lengths
