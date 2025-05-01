# Chunker Application Business Logic

This document outlines the business logic of the Chunker application, which is designed to process and analyze video content, breaking it down into meaningful segments for detailed analysis.

## Core Workflow

```mermaid
flowchart TD
    Start([Start]) --> InputVideo[Process Video Input]
    InputVideo --> SplitSegments[Split Video into Segments]
    SplitSegments --> ParallelProcessing{Parallel Processing}
    ParallelProcessing --> |Segment 1| ProcessSegment1[Process Segment]
    ParallelProcessing --> |Segment 2| ProcessSegment2[Process Segment]
    ParallelProcessing --> |Segment N| ProcessSegmentN[Process Segment]

    ProcessSegment1 --> |Results| CombineResults[Combine Results]
    ProcessSegment2 --> |Results| CombineResults
    ProcessSegmentN --> |Results| CombineResults

    CombineResults --> FullVideoAnalysis[Create Full Video Analysis]
    FullVideoAnalysis --> VectorDB[(Store in Vector DB)]
    VectorDB --> ChatbotInterface[Chatbot Interface]
    ChatbotInterface --> End([End])

    subgraph "Per Segment Processing"
        ProcessSegment[Process Segment] --> ExtractFrames[Extract Video Frames]
        ExtractFrames --> TranscribeAudio[Transcribe Audio]
        TranscribeAudio --> AnalyzeFrames[Analyze Frames with AI]
        AnalyzeFrames --> StructuredData[Extract Structured Data]
        StructuredData --> StoreResults[Store Analysis Results]
    end
```

## Data Model

```mermaid
classDiagram
    VideoAnalysis "1" -- "0..*" Person
    VideoAnalysis "1" -- "1" VisualDetails
    VideoAnalysis "1" -- "1" ContentAnalysis
    VideoAnalysis "1" -- "0..1" TechnicalDetails
    VideoAnalysis "1" -- "0..1" AudienceAnalysis
    VideoAnalysis "1" -- "0..1" BrandAnalysis

    class VideoAnalysis {
        +String video_name
        +String segment_timeframe
        +PromptType analysis_type
        +String text_analysis
        +List~Person~ people
        +VisualDetails visuals
        +ContentAnalysis content_details
        +TechnicalDetails technical_details
        +AudienceAnalysis audience_analysis
        +BrandAnalysis brand_analysis
        +String call_to_action
        +List~String~ key_messages
        +String transcription
    }

    class Person {
        +String role
        +String description
        +List~String~ actions
        +String emotional_tone
    }

    class VisualDetails {
        +List~String~ composition
        +String camera_movement
        +String lighting
        +String color_scheme
    }

    class ContentAnalysis {
        +String setting
        +String platform
        +String message_purpose
        +String emotional_appeal
    }

    class TechnicalDetails {
        +String production_quality
        +String platform_optimization
        +String text_elements
        +List~String~ notable_techniques
    }

    class AudienceAnalysis {
        +List~String~ target_demographics
        +List~String~ pain_points
        +List~String~ appeal_elements
    }

    class BrandAnalysis {
        +List~String~ brand_elements
        +String brand_voice
        +String consistency
    }
```

## Analysis Process

```mermaid
sequenceDiagram
    participant User
    participant Main
    participant Analyzer
    participant VideoProcessor
    participant AudioProcessor
    participant AIClient
    participant DBManager
    participant Chatbot

    User->>Main: Run with video path
    Main->>VideoProcessor: Get video metadata
    VideoProcessor-->>Main: Video duration & metadata
    Main->>Main: Calculate segments

    loop For each segment
        Main->>Analyzer: Process segment
        Analyzer->>VideoProcessor: Extract frames
        VideoProcessor-->>Analyzer: Base64 frames

        Analyzer->>AudioProcessor: Process audio
        AudioProcessor-->>Analyzer: Transcription

        Analyzer->>AIClient: Analyze video frames
        AIClient-->>Analyzer: Text analysis

        Analyzer->>AIClient: Extract structured data
        AIClient-->>Analyzer: Structured VideoAnalysis

        Analyzer-->>Main: Analysis results

        Main->>DBManager: Store analysis
        DBManager->>DBManager: Generate embedding
        DBManager-->>Main: Stored document ID
    end

    Main->>Analyzer: Create full video analysis
    Analyzer->>VideoProcessor: Extract frames (sparse)
    VideoProcessor-->>Analyzer: Base64 frames
    Analyzer->>AudioProcessor: Process full audio
    AudioProcessor-->>Analyzer: Full transcription
    Analyzer->>AIClient: Analyze full video
    AIClient-->>Analyzer: Text analysis
    Analyzer-->>Main: Full analysis

    Main->>DBManager: Store full analysis
    DBManager-->>Main: Stored document ID

    opt If chat enabled
        Main->>Chatbot: Start chatbot
        User->>Chatbot: Query about video
        Chatbot->>DBManager: Vector search
        DBManager-->>Chatbot: Relevant segments
        Chatbot->>AIClient: Generate response
        AIClient-->>Chatbot: Response
        Chatbot-->>User: Answer
    end
```

## Component Architecture

```mermaid
flowchart TB
    User((User)) --> CLI
    CLI --> Main

    Main --> Analyzer
    Main --> DBManager
    Main --> EmbeddingsManager
    Main --> VideoChatbot

    Analyzer --> VideoProcessor
    Analyzer --> AudioProcessor
    Analyzer --> AIClient

    VideoChatbot --> DBManager
    VideoChatbot --> AIClient

    DBManager --> VectorDB[(Vector Database)]
    EmbeddingsManager --> AIClient

    subgraph Core Components
        Main[main.py]
        Analyzer[analyzer.py]
        DBManager[db_manager.py]
        EmbeddingsManager[embeddings_manager.py]
        VideoChatbot[chatbot.py]
    end

    subgraph Processing Services
        VideoProcessor[video_processor.py]
        AudioProcessor[audio_processor.py]
        AIClient[OpenAI API Client]
    end

    subgraph Data Models
        Models[models.py]
        Config[config.py]
    end

    Main --> Models
    Main --> Config
    Analyzer --> Models
    Analyzer --> Config
```

## Vector Database Flow

```mermaid
flowchart TD
    VideoSegments[Video Segments Analysis] --> |Store| EmbeddingProcess
    EmbeddingProcess[Generate Embeddings] --> VectorDB[(Vector Database)]

    User([User]) --> |Query| Chatbot
    Chatbot[Video Chatbot] --> |Search| VectorDB
    VectorDB --> |Semantic Results| Chatbot
    Chatbot --> |Generate Response| AIModel[AI Model]
    AIModel --> |Response| User

    subgraph Embedding Process
        Analysis[Analysis Text] --> TextProcessing[Text Processing]
        TextProcessing --> EmbeddingGeneration[Embedding Generation]
        EmbeddingGeneration --> DBStorage[Store in DB]
    end
```

## PromptType Enum Flow

```mermaid
flowchart LR
    PromptType --> |General| General[General Analysis]
    PromptType --> |Audience| Audience[Audience Analysis]
    PromptType --> |Visual| Visual[Visual Analysis]
    PromptType --> |Technical| Technical[Technical Analysis]
    PromptType --> |Brand| Brand[Brand Analysis]

    General --> Analyzer
    Audience --> Analyzer
    Visual --> Analyzer
    Technical --> Analyzer
    Brand --> Analyzer

    Analyzer --> VideoAnalysis
```
