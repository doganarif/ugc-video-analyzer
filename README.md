# Social Media & UGC Video Analyzer

A comprehensive tool for analyzing social media ads and user-generated content (UGC) videos using Azure OpenAI's GPT-4o, providing insights on marketing effectiveness, audience targeting, visual branding, and more. Now with vector database integration for RAG-based AI chatbot capabilities.

[![Youtube Video](https://img.youtube.com/vi/doICR6wBOLU/0.jpg)](https://www.youtube.com/watch?v=doICR6wBOLU)


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
- **NEW:** Vector database (PostgreSQL with pgvector) integration
- **NEW:** Asynchronous video processing for faster analysis
- **NEW:** CLI chatbot for interacting with analyzed videos

## Documentation

- [LOGIC.md](LOGIC.md) - Contains Mermaid diagrams visualizing the application's business logic, component architecture, and data flow

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install PostgreSQL and pgvector extension:

   ```
   # For Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib

   # For macOS with Homebrew
   brew install postgresql

   # Install pgvector extension (requires PostgreSQL 11+)
   # Follow instructions at: https://github.com/pgvector/pgvector
   ```

4. Create a database for the video analytics:
   ```
   createdb video_analytics
   ```
5. Create a `.env` file based on the provided `.env.example` with your API credentials and database connection:
   ```
   cp .env.example .env
   # Edit .env with your credentials
   ```
   Example `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   DATABASE_URL=postgresql://username:password@localhost:5432/video_analytics
   CHAT_MODEL=gpt-4o
   ```

## Usage

### Async Video Analysis with Vector DB

Process a video with asynchronous analysis and store in vector database:

```bash
python main.py --video your_video.mp4 --interval 5 --analysis-types general audience visual
```

Options:

- `--video, -v`: Path to the video file (required)
- `--output, -o`: Output directory for analysis (default: "analysis_output")
- `--interval, -i`: Interval in seconds for segmenting the video (default: 10)
- `--chat, -c`: Start chatbot after analysis
- `--analysis-types, -a`: Types of analysis to perform (default: general audience visual)

### AI Chatbot for Video Q&A

After analyzing videos, you can interact with them using the chatbot:

```bash
python chat.py --video your_video.mp4
```

This will start a CLI chatbot that allows you to ask questions about the video content.

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

### Comprehensive Analysis with Vector DB

Asynchronously analyze a video and store in vector database:

```python
import asyncio
from main import async_comprehensive_analysis
from config import PromptType

results = asyncio.run(async_comprehensive_analysis(
    video_path='your_ad.mp4',
    output_dir='analysis_output',
    analysis_types=[PromptType.GENERAL, PromptType.AUDIENCE, PromptType.VISUAL],
    shot_interval=5  # 5-second segments
))
```

## Project Structure

- `config.py` - Configuration parameters, constants, and prompt templates
- `models.py` - Pydantic models for structured data
- `video_processor.py` - Video processing functions
- `audio_processor.py` - Audio transcription functions
- `analyzer.py` - Core analysis logic
- `main.py` - Main execution and async video processing
- `db_manager.py` - PostgreSQL with pgvector integration
- `embeddings_manager.py` - Text embedding generation
- `chatbot.py` - RAG-based AI chatbot for video Q&A
- `chat.py` - CLI script for starting the chatbot
- `requirements.txt` - Project dependencies
- `LOGIC.md` - Business logic diagrams and application architecture

## Example

Run the example script with chatbot:

```bash
python main.py --video sample.mp4 --interval 5 --chat
```

This will:

1. Analyze the sample video in 5-second segments
2. Store analysis in the PostgreSQL vector database
3. Start the chatbot for interactive Q&A about the video

## Business Logic

The application follows a structured workflow:

1. Video is split into segments for parallel processing
2. Each segment undergoes frame extraction, audio transcription, and AI analysis
3. Results are structured and stored in a vector database
4. A full video analysis is performed and stored
5. Optional chatbot interface leverages this data for user queries

For detailed visualizations of the business logic, data models, and component architecture, see [LOGIC.md](LOGIC.md).

## Notes

- This tool requires OpenAI API access (or Azure OpenAI services)
- Processing longer videos can be expensive in terms of API costs
- Default settings are optimized for short-form content (15-60 seconds)
- Frame extraction rate and segment intervals can be adjusted for different video lengths
- PostgreSQL with pgvector extension is required for vector database functionality

## Contributing

### Code Structure

This project follows a modular approach, with separate modules for:

1. **Processing** - Video and audio processing functionality (`video_processor.py`, `audio_processor.py`)
2. **Analysis** - Core analysis logic (`analyzer.py`)
3. **Database** - Data storage and vector search (`db_manager.py`, `embeddings_manager.py`)
4. **User Interface** - CLI and chatbot functionality (`chat.py`, `chatbot.py`)
5. **Configuration** - Settings and prompts (`config.py`)
6. **Data Models** - Pydantic models for type safety (`models.py`)

### Coding Standards

Please follow these standards when contributing:

1. **Type Hints** - Use Python type hints for all function parameters and return values
2. **Documentation** - Add docstrings to all functions and classes
3. **Error Handling** - Use appropriate exception handling
4. **Logging** - Use the logging module instead of print statements
5. **Factory Functions** - Use factory functions in `config.py` for client initialization
6. **PEP 8** - Follow PEP 8 style guidelines

### Development Setup

1. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install development dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up pre-commit hooks:
   ```
   pip install pre-commit
   pre-commit install
   ```

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding standards
4. Test your changes
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
