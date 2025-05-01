import cv2
import os
import time
import json
import re
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from openai import AzureOpenAI
import base64
from pydantic import BaseModel, Field

# Default configuration
DEFAULT_SHOT_INTERVAL = 30  # In seconds
DEFAULT_FRAMES_PER_SECOND = 1
DEFAULT_TEMPERATURE = 0.5
RESIZE_OF_FRAMES = 4

# Define prompts as an enum for easier selection
class PromptType(str, Enum):
    GENERAL = "general"
    CHARACTER = "character"
    VISUAL = "visual"
    NARRATIVE = "narrative"
    CULTURAL = "cultural"
    DIRECTOR = "director"
    TECHNICAL = "technical"
    SCENE = "scene"

# System prompts dictionary
SYSTEM_PROMPTS = {
    PromptType.GENERAL: """You are an expert film analyst with deep knowledge of cinematography, narrative structure, and visual storytelling. Analyze the frames from this movie sequence and provide:
1. Scene setting and context
2. Key characters present and their actions
3. Cinematography techniques (shot types, lighting, color palette)
4. Narrative progression in this segment
5. Emotional tone and atmosphere
Be specific about visual details that reveal the filmmaker's intent and storytelling approach.""",

    PromptType.CHARACTER: """You are a character analysis specialist for film. From these sequential frames, identify the main characters present and analyze:
1. Their physical appearance and expressions
2. Body language and interactions with others
3. Character development evident in this sequence
4. Performance nuances by the actors
5. How the cinematography frames or emphasizes certain characters
Provide insightful observations about character dynamics and emotional states.""",

    PromptType.VISUAL: """You are a cinematography expert specializing in visual film analysis. From these movie frames, provide detailed observations about:
1. Shot composition, framing and camera movement
2. Lighting techniques and their emotional impact
3. Color grading and palette choices
4. Visual motifs and symbolism
5. Set design and visual environment
Explain how these visual elements contribute to the storytelling and mood of the scene.""",

    PromptType.NARRATIVE: """You are a film narrative expert. Analyze these sequential frames to identify:
1. Where this sequence likely fits in the three-act structure
2. Plot developments occurring in this segment
3. Narrative techniques being employed
4. Foreshadowing or setup elements
5. How this sequence likely connects to broader story arcs
Provide context for how this scene functions within the overall narrative framework.""",

    PromptType.CULTURAL: """You are a film genre and cultural analysis specialist. From these movie frames, identify:
1. The apparent genre(s) and their defining visual elements
2. Cultural contexts and references
3. Time period indicators and historical elements
4. Genre conventions or subversions present
5. Cultural themes or commentary being expressed
Analyze how the visuals communicate within specific genre traditions and cultural frameworks.""",

    PromptType.DIRECTOR: """You are a film auteur specialist who recognizes distinctive directorial styles. From these movie frames, analyze:
1. Characteristic visual techniques that might identify the director
2. Signature shot compositions, camera movements, or editing patterns
3. Thematic elements common to particular directors
4. Visual or narrative motifs that appear director-specific
5. Stylistic comparisons to known directors if the creator is unidentified
Provide detailed analysis of how directorial choices shape the viewing experience and compare to established filmmaking styles.""",

    PromptType.TECHNICAL: """You are a technical filmmaking expert. From these movie frames, provide precise analysis of:
1. Camera techniques (lens choice, movement, positioning)
2. Lighting setups and design (key light, fill, practical sources)
3. Production design elements (set construction, props, visual authenticity)
4. Editing patterns visible from frame sequences
5. Technical execution quality and production value
6. Special effects or VFX implementation
Explain how these technical choices support storytelling and what equipment or techniques were likely used to achieve these results.""",

    PromptType.SCENE: """You are a scene analysis specialist. Examine these sequential frames and provide:
1. Beat-by-beat breakdown of the scene's progression
2. Purpose of this scene in the broader narrative context
3. Character objectives and conflicts
4. Emotional beats and tension points
5. Scene structure and pacing
Explain how this scene is constructed to achieve specific dramatic or narrative goals."""
}

# Default user prompt
USER_PROMPT = "These are the frames from the video. Analyze them according to your expertise."

# Pydantic models for structured output
class Character(BaseModel):
    name: str = Field(..., description="Character name or descriptor if name is unknown")
    description: str = Field(..., description="Physical description of the character")
    actions: List[str] = Field(default_factory=list, description="Key actions performed by the character")
    emotional_state: Optional[str] = Field(None, description="Emotional state of the character")

class CinematographyDetails(BaseModel):
    shot_types: List[str] = Field(default_factory=list, description="Types of shots used (close-up, medium, wide, etc.)")
    camera_movement: Optional[str] = Field(None, description="Description of camera movements")
    lighting: Optional[str] = Field(None, description="Lighting style and techniques")
    color_palette: Optional[str] = Field(None, description="Description of color palette and significance")

class SceneAnalysis(BaseModel):
    setting: str = Field(..., description="Location and setting of the scene")
    time_period: Optional[str] = Field(None, description="Apparent time period of the scene")
    narrative_purpose: Optional[str] = Field(None, description="Purpose of this scene in the narrative")
    emotional_tone: Optional[str] = Field(None, description="Overall emotional tone of the scene")

class TechnicalDetails(BaseModel):
    camera_techniques: Optional[str] = Field(None, description="Camera techniques utilized")
    production_design: Optional[str] = Field(None, description="Set design and production elements")
    special_effects: Optional[str] = Field(None, description="Special effects or VFX used")
    notable_techniques: List[str] = Field(default_factory=list, description="Other notable technical aspects")

class GenreAnalysis(BaseModel):
    genres: List[str] = Field(default_factory=list, description="Apparent genres of the film")
    genre_elements: List[str] = Field(default_factory=list, description="Visual elements indicating genre")
    cultural_references: List[str] = Field(default_factory=list, description="Cultural references or contexts")

class DirectorStyle(BaseModel):
    style_elements: List[str] = Field(default_factory=list, description="Notable stylistic elements")
    possible_directors: List[str] = Field(default_factory=list, description="Possible directors with similar style")
    signature_techniques: List[str] = Field(default_factory=list, description="Signature techniques visible")

class MovieAnalysis(BaseModel):
    video_name: str = Field(..., description="Name of the analyzed video file")
    segment_timeframe: Optional[str] = Field(None, description="Timeframe of the analyzed segment")
    analysis_type: PromptType = Field(..., description="Type of analysis performed")
    text_analysis: str = Field(..., description="Full text analysis from the AI")
    characters: List[Character] = Field(default_factory=list, description="Characters identified in the scene")
    cinematography: CinematographyDetails = Field(default_factory=CinematographyDetails, description="Cinematography analysis")
    scene_details: SceneAnalysis = Field(default_factory=SceneAnalysis, description="Scene analysis details")
    technical_details: Optional[TechnicalDetails] = Field(None, description="Technical filmmaking details")
    genre_analysis: Optional[GenreAnalysis] = Field(None, description="Genre and cultural analysis")
    director_style: Optional[DirectorStyle] = Field(None, description="Director style analysis")
    transcription: Optional[str] = Field(None, description="Audio transcription if available")

    class Config:
        json_schema_extra = {
            "example": {
                "video_name": "movie_clip.mp4",
                "segment_timeframe": "00:05:20 - 00:07:45",
                "analysis_type": "general",
                "text_analysis": "This scene takes place in a dimly lit urban setting...",
                "characters": [
                    {
                        "name": "Main protagonist",
                        "description": "Middle-aged man in a dark suit",
                        "actions": ["Walking cautiously", "Looking over shoulder"],
                        "emotional_state": "Anxious and vigilant"
                    }
                ],
                "cinematography": {
                    "shot_types": ["Close-up", "Dutch angle"],
                    "camera_movement": "Handheld tracking shot",
                    "lighting": "Low-key lighting with strong shadows",
                    "color_palette": "Desaturated blues and greens"
                },
                "scene_details": {
                    "setting": "Urban alleyway at night",
                    "time_period": "Contemporary",
                    "narrative_purpose": "Building tension before confrontation",
                    "emotional_tone": "Suspenseful and ominous"
                }
            }
        }

# Load configuration
load_dotenv(override=True)

# Configuration of OpenAI GPT-4o
aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_apikey = os.environ["AZURE_OPENAI_API_KEY"]
aoai_apiversion = os.environ["AZURE_OPENAI_API_VERSION"]
aoai_model_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]

# Create AOAI client for answer generation
aoai_client = AzureOpenAI(
    azure_deployment=aoai_model_name,
    api_version=aoai_apiversion,
    azure_endpoint=aoai_endpoint,
    api_key=aoai_apikey
)

# Configuration of Whisper
whisper_endpoint = os.environ["WHISPER_ENDPOINT"]
whisper_apikey = os.environ["WHISPER_API_KEY"]
whisper_apiversion = os.environ["WHISPER_API_VERSION"]
whisper_model_name = os.environ["WHISPER_DEPLOYMENT_NAME"]

# Create AOAI client for whisper
whisper_client = AzureOpenAI(
    api_version=whisper_apiversion,
    azure_endpoint=whisper_endpoint,
    api_key=whisper_apikey
)

# Function to encode a local video into frames
def process_video(video_path, frames_per_second=DEFAULT_FRAMES_PER_SECOND, resize=RESIZE_OF_FRAMES, output_dir='',
                  temperature=DEFAULT_TEMPERATURE, max_frames=100):
    print(f"Starting video processing for {video_path} with frames_per_second={frames_per_second}, resize={resize}, max_frames={max_frames}")
    base64Frames = []

    # Prepare the video analysis
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # Calculate the appropriate frames to skip based on max_frames
    # This ensures we don't exceed API limits even for long videos
    if max_frames > 0:
        effective_fps = max_frames / duration
        # Use the lower of the two rates to avoid exceeding max_frames
        effective_fps = min(effective_fps, frames_per_second)
        frames_to_skip = int(fps / effective_fps)
    else:
        frames_to_skip = int(fps / frames_per_second)

    curr_frame = 0
    frame_count = 1

    # Prepare to write the frames to disk
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)

    # Loop through the video and extract frames at the specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break

        print(f"Processing frame {curr_frame}/{total_frames} ({frame_count}/{max_frames if max_frames > 0 else 'unlimited'})")

        # Resize the frame if required
        if resize != 0:
            height, width, _ = frame.shape
            frame = cv2.resize(frame, (width // resize, height // resize))

        _, buffer = cv2.imencode(".jpg", frame)

        # Save frame as JPG file if output_dir is specified
        if output_dir != '':
            frame_filename = os.path.join(output_dir,
                                          f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg")
            with open(frame_filename, "wb") as f:
                f.write(buffer)

        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
        frame_count += 1

        # Stop if we've reached the maximum number of frames
        if max_frames > 0 and len(base64Frames) >= max_frames:
            print(f"Reached maximum frames limit ({max_frames})")
            break

    video.release()
    print(f"Extracted {len(base64Frames)} frames from {video_path}")

    return base64Frames

# Function to transcript the audio from the local video with Whisper
def process_audio(video_path):
    print(f"Starting audio transcription for {video_path}")
    transcription_text = ''
    try:
        base_video_path, _ = os.path.splitext(video_path)
        audio_path = f"{base_video_path}.mp3"
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, bitrate="32k")
        clip.audio.close()
        clip.close()
        print(f"Extracted audio to {audio_path}")

        # Transcribe the audio
        print(f"Transcribing audio from {audio_path}")
        transcription = whisper_client.audio.transcriptions.create(
            model=whisper_model_name,
            file=open(audio_path, "rb"),
        )
        transcription_text = transcription.text
        print("Transcript: ", transcription_text + "\n\n")
    except Exception as ex:
        print(f'ERROR Transcript: {ex}')
        transcription_text = ''

    return transcription_text

# Function to analyze the video with GPT-4o
def analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature):
    print(f"Starting video analysis with system_prompt={system_prompt} and user_prompt={user_prompt}")
    print(f"Number of frames to analyze: {len(base64frames)}")

    # Maximum number of images per request to avoid API limitations
    MAX_IMAGES_PER_REQUEST = 45  # Setting to 45 to be safe (below 50 limit)

    # Initialize combined response
    combined_response = ""

    # Split frames into batches
    num_batches = (len(base64frames) + MAX_IMAGES_PER_REQUEST - 1) // MAX_IMAGES_PER_REQUEST
    print(f"Splitting frames into {num_batches} batches")

    for batch_idx in range(num_batches):
        batch_start = batch_idx * MAX_IMAGES_PER_REQUEST
        batch_end = min((batch_idx + 1) * MAX_IMAGES_PER_REQUEST, len(base64frames))
        batch_frames = base64frames[batch_start:batch_end]

        print(f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_frames)} frames")

        # Adjust the prompt for batch context
        batch_prompt = user_prompt
        if num_batches > 1:
            batch_prompt = f"{user_prompt} (Analyzing frames {batch_start+1} to {batch_end} of {len(base64frames)})"

        try:
            if transcription and batch_idx == 0:  # Include the audio transcription only in the first batch
                response = aoai_client.chat.completions.create(
                    model=aoai_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": batch_prompt},
                        {"role": "user", "content": [
                            *map(lambda x: {"type": "image_url",
                                            "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}},
                                 batch_frames),
                            {"type": "text",
                             "text": f"The audio transcription is: {transcription if isinstance(transcription, str) else transcription.text}"}
                        ]}
                    ],
                    temperature=temperature,
                    max_tokens=4096
                )
            else:  # Without the audio transcription or subsequent batches
                response = aoai_client.chat.completions.create(
                    model=aoai_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": batch_prompt},
                        {"role": "user", "content": [
                            *map(lambda x: {"type": "image_url",
                                            "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}},
                                 batch_frames),
                        ]}
                    ],
                    temperature=temperature,
                    max_tokens=4096
                )

            json_response = json.loads(response.model_dump_json())
            batch_response = json_response['choices'][0]['message']['content']

            # Append to combined response
            if batch_idx > 0:
                combined_response += "\n\n--- CONTINUED ANALYSIS ---\n\n"
            combined_response += batch_response

            print(f"Batch {batch_idx + 1}/{num_batches} analyzed successfully")

        except Exception as ex:
            print(f'ERROR Video Analyzer Batch {batch_idx + 1}: {ex}')
            combined_response += f"\n\nERROR in Batch {batch_idx + 1}: {ex}"

    # If we did multiple batches, do a final consolidation analysis
    if num_batches > 1:
        try:
            print("Performing final consolidation analysis...")
            consolidation_prompt = f"""
            The following is a multi-part analysis of a video, broken into {num_batches} sequential segments due to technical constraints.
            Please synthesize this into a single coherent analysis, removing redundancies and creating a comprehensive overview.
            Focus on the overall narrative, character development, and filmmaking techniques across the entire sequence.
            """

            consolidation_response = aoai_client.chat.completions.create(
                model=aoai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": consolidation_prompt},
                    {"role": "user", "content": combined_response}
                ],
                temperature=temperature,
                max_tokens=4096
            )

            json_consolidation = json.loads(consolidation_response.model_dump_json())
            final_response = json_consolidation['choices'][0]['message']['content']
            print("Final consolidation analysis completed successfully")
            return final_response
        except Exception as ex:
            print(f'ERROR in consolidation analysis: {ex}')
            # Return the concatenated analysis if consolidation fails
            return combined_response

    return combined_response

# Function to extract structured data from text analysis
def extract_structured_data(text_analysis, analysis_type, video_name, segment_timeframe=None, transcription=None):
    """
    Extract structured data from text analysis using GPT-4o
    """
    print(f"Extracting structured data from text analysis for {video_name}")

    # Define the system prompt for extracting structured data
    system_prompt = f"""
    You are an expert in structured data extraction from movie analysis text. 
    Your task is to carefully read the provided movie analysis text and extract specific information
    to create a detailed JSON object.
    
    The analysis type is: {analysis_type.value}
    
    VERY IMPORTANT: You must extract as much information as possible from the text and populate the JSON fields
    accordingly. Do not return null or empty lists unless that information is truly absent from the analysis.
    
    If the analysis text doesn't explicitly mention something but reasonably implies it, make an educated inference
    to fill in those fields rather than leaving them empty.
    
    Here are the JSON fields you need to populate:
    
    1. video_name: Use the provided video name
    2. segment_timeframe: Use the provided timeframe
    3. analysis_type: Use the provided analysis type
    4. text_analysis: Use the full provided text analysis
    5. characters: Extract any mentioned characters, each with:
       - name: Character name or descriptor
       - description: Physical description
       - actions: List of key actions
       - emotional_state: Emotional state if mentioned
    
    6. cinematography: Details about:
       - shot_types: List of shot types mentioned (close-up, wide, etc.)
       - camera_movement: Any camera movement described
       - lighting: Lighting style and techniques mentioned
       - color_palette: Color palette description
    
    7. scene_details:
       - setting: Location and setting of the scene (REQUIRED)
       - time_period: Apparent time period
       - narrative_purpose: Purpose of the scene in the narrative
       - emotional_tone: Overall emotional tone
    
    8. technical_details:
       - camera_techniques: Camera techniques mentioned
       - production_design: Set design and production elements
       - special_effects: Special effects or VFX mentioned
       - notable_techniques: Other notable technical aspects
    
    9. genre_analysis:
       - genres: Apparent genres
       - genre_elements: Visual elements indicating genre
       - cultural_references: Cultural references or contexts
    
    10. director_style:
        - style_elements: Notable stylistic elements
        - possible_directors: Possible directors with similar style
        - signature_techniques: Signature techniques visible
    
    11. transcription: Use the provided transcription
    
    For fields that aren't explicitly mentioned in the text but can be reasonably inferred, 
    make appropriate inferences rather than leaving them empty.
    """

    try:
        response = aoai_client.chat.completions.create(
            model=aoai_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analysis text: {text_analysis}"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=4096
        )

        structured_data = json.loads(response.choices[0].message.content)

        # Add the metadata that we know for sure
        structured_data["video_name"] = video_name
        structured_data["segment_timeframe"] = segment_timeframe
        structured_data["analysis_type"] = analysis_type.value
        structured_data["text_analysis"] = text_analysis
        structured_data["transcription"] = transcription

        # Fix potential type mismatches - convert lists to strings where needed
        if "cinematography" in structured_data and structured_data["cinematography"]:
            if "camera_movement" in structured_data["cinematography"]:
                if isinstance(structured_data["cinematography"]["camera_movement"], list):
                    structured_data["cinematography"]["camera_movement"] = ", ".join(structured_data["cinematography"]["camera_movement"]) if structured_data["cinematography"]["camera_movement"] else None

        if "technical_details" in structured_data and structured_data["technical_details"]:
            if "camera_techniques" in structured_data["technical_details"]:
                if isinstance(structured_data["technical_details"]["camera_techniques"], list):
                    structured_data["technical_details"]["camera_techniques"] = ", ".join(structured_data["technical_details"]["camera_techniques"]) if structured_data["technical_details"]["camera_techniques"] else None
            
            if "special_effects" in structured_data["technical_details"]:
                if isinstance(structured_data["technical_details"]["special_effects"], list):
                    structured_data["technical_details"]["special_effects"] = ", ".join(structured_data["technical_details"]["special_effects"]) if structured_data["technical_details"]["special_effects"] else None
            
            if "production_design" in structured_data["technical_details"]:
                if isinstance(structured_data["technical_details"]["production_design"], list):
                    structured_data["technical_details"]["production_design"] = ", ".join(structured_data["technical_details"]["production_design"]) if structured_data["technical_details"]["production_design"] else None

        # Ensure setting is not empty (required field)
        if "scene_details" not in structured_data or not structured_data["scene_details"]:
            structured_data["scene_details"] = {"setting": "Opening credits/title sequence"}
        elif "setting" not in structured_data["scene_details"] or not structured_data["scene_details"]["setting"]:
            structured_data["scene_details"]["setting"] = "Setting implied from analysis"

        # Try to extract setting from text if missing
        if structured_data.get("scene_details", {}).get("setting") == "Unknown setting" or structured_data.get("scene_details", {}).get("setting") == "Error extracting data, see text_analysis field":
            # Simple extraction of potential setting descriptions
            setting_indicators = ["takes place in", "set in", "located in", "setting is", "background shows", "unfolds in", "environment is"]
            extracted_setting = None
            
            for indicator in setting_indicators:
                if indicator in text_analysis.lower():
                    # Extract the sentence containing the indicator
                    sentences = text_analysis.split('.')
                    for sentence in sentences:
                        if indicator in sentence.lower():
                            extracted_setting = sentence.strip() + "."
                            break
                    if extracted_setting:
                        break
            
            # If we didn't find a setting using indicators, look for first paragraph which often contains setting info
            if not extracted_setting and text_analysis:
                paragraphs = text_analysis.split('\n\n')
                if paragraphs:
                    first_paragraph = paragraphs[0].strip()
                    # Use first sentence if paragraph is long
                    if len(first_paragraph) > 100:
                        first_sentence = first_paragraph.split('.')[0] + '.'
                        extracted_setting = first_sentence
                    else:
                        extracted_setting = first_paragraph
            
            # Use the extracted setting if we found one
            if extracted_setting:
                if "scene_details" not in structured_data:
                    structured_data["scene_details"] = {}
                structured_data["scene_details"]["setting"] = extracted_setting

        # Validate with Pydantic
        try:
            movie_analysis = MovieAnalysis(**structured_data)
            return movie_analysis
        except Exception as validation_error:
            print(f"Validation error: {validation_error}")
            # If validation fails, try to fix common issues
            if "scene_details" not in structured_data or not structured_data["scene_details"]:
                structured_data["scene_details"] = {"setting": "Unknown setting"}

            # Ensure all required fields have at least default values
            if "cinematography" not in structured_data or not structured_data["cinematography"]:
                structured_data["cinematography"] = {}
            if "characters" not in structured_data:
                structured_data["characters"] = []

            # Try again with fixed data
            movie_analysis = MovieAnalysis(**structured_data)
            return movie_analysis

    except Exception as ex:
        print(f'ERROR extracting structured data: {ex}')
        # Create a minimal valid object as fallback
        return MovieAnalysis(
            video_name=video_name,
            analysis_type=analysis_type,
            text_analysis=text_analysis,
            scene_details=SceneAnalysis(setting="Error extracting data, see text_analysis field"),
            transcription=transcription
        )

# Split the video into shots of N seconds
def split_video(video_path, output_dir, shot_interval=DEFAULT_SHOT_INTERVAL, max_duration=None):
    """
    Split the video into shots of specified duration

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the split shots
        shot_interval: Duration of each shot in seconds
        max_duration: Maximum duration to process in seconds

    Returns:
        Generator yielding (output_file_path, timeframe_string) tuples
    """
    print(f"Starting video splitting for {video_path} with shot_interval={shot_interval}, max_duration={max_duration}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    if max_duration is not None and max_duration > 0:
        duration = min(duration, max_duration)

    os.makedirs(output_dir, exist_ok=True)

    for start_time in range(0, int(duration), shot_interval):
        end_time = min(start_time + shot_interval, duration)
        output_file = os.path.join(output_dir,
                                   f'{os.path.splitext(os.path.basename(video_path))[0]}_shot_{start_time}-{end_time}_secs.mp4')
        print(f"Extracting shot from {start_time} to {end_time} into {output_file}")

        try:
            ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_file)
            timeframe = f"{start_time}-{end_time}"
            yield output_file, timeframe
        except Exception as ex:
            print(f"Error extracting shot {start_time}-{end_time}: {ex}")
            # Continue with the next shot

# Process the video with specified analysis type
def execute_video_processing(shot_path, analysis_type, user_prompt=USER_PROMPT, temperature=DEFAULT_TEMPERATURE,
                             frames_per_second=DEFAULT_FRAMES_PER_SECOND, analysis_dir='analysis_output',
                             save_frames=False, audio_transcription=False, resize=RESIZE_OF_FRAMES,
                             segment_timeframe=None, max_frames=80):
    print(f"Starting video processing for shot {shot_path} with analysis_type={analysis_type.value}")

    # Ensure analysis directory exists
    os.makedirs(analysis_dir, exist_ok=True)

    # Get the appropriate system prompt
    system_prompt = SYSTEM_PROMPTS[analysis_type]

    # Extract frames at the specified rate
    print(f"Extracting frames from {shot_path}")
    start_time = time.time()
    if save_frames:
        output_dir = os.path.join(analysis_dir, 'frames')
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = ''

    base64frames = process_video(
        shot_path,
        frames_per_second=frames_per_second,
        resize=resize,
        output_dir=output_dir,
        temperature=temperature,
        max_frames=max_frames  # Limit maximum frames to avoid API issues
    )
    end_time = time.time()
    print(f'\t>>>> Frames extraction took {(end_time - start_time):.3f} seconds <<<<')

    # Extract the transcription of the audio
    transcription = ''
    if audio_transcription:
        print(f"Transcribing audio from {shot_path}")
        start_time = time.time()
        transcription = process_audio(shot_path)
        end_time = time.time()
        print(f'Transcription: [{transcription}]')
        print(f'\t>>>> Audio transcription took {(end_time - start_time):.3f} seconds <<<<')
    else:
        print(f"Skipping audio transcription")

    # Analyze the video frames and the audio transcription with GPT-4o
    print(f"Analyzing frames with {aoai_model_name} using {analysis_type.value} prompt")
    print(f"Processing {len(base64frames)} frames in batches")
    start_time = time.time()
    text_analysis = analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature)
    end_time = time.time()
    print(f'\t>>>> Analysis with {aoai_model_name} took {(end_time - start_time):.3f} seconds <<<<')

    # Extract structured data from the text analysis
    print(f"Extracting structured data from text analysis")
    start_time = time.time()
    structured_data = extract_structured_data(text_analysis, analysis_type,
                                              os.path.basename(shot_path),
                                              segment_timeframe, transcription)
    end_time = time.time()
    print(f'\t>>>> Structured data extraction took {(end_time - start_time):.3f} seconds <<<<')

    # Save the analysis to a JSON file in the analysis directory
    analysis_filename = os.path.join(analysis_dir,
                                     f"{os.path.splitext(os.path.basename(shot_path))[0]}_{analysis_type.value}_analysis.json")
    with open(analysis_filename, 'w') as json_file:
        json.dump(structured_data.model_dump(), json_file, indent=4)
    print(f"Analysis saved as: {analysis_filename}")

    # Also save raw text analysis
    text_analysis_filename = os.path.join(analysis_dir,
                                          f"{os.path.splitext(os.path.basename(shot_path))[0]}_{analysis_type.value}_text.txt")
    with open(text_analysis_filename, 'w') as f:
        f.write(text_analysis)
    print(f"Raw text analysis saved as: {text_analysis_filename}")

    return structured_data

def analyze_movie_with_multiple_perspectives(video_path, output_dir='analysis_output', shot_interval=30,
                                             max_duration=None, frames_per_second=DEFAULT_FRAMES_PER_SECOND,
                                             temperature=DEFAULT_TEMPERATURE, save_frames=False,
                                             audio_transcription=True, resize=RESIZE_OF_FRAMES,
                                             analysis_types=None, max_frames=80):
    """
    Analyze a movie from multiple perspectives by using different prompt types

    Args:
        video_path: Path to the video file
        output_dir: Directory to save analysis results
        shot_interval: Interval in seconds to split the video
        max_duration: Maximum duration to analyze (in seconds)
        frames_per_second: Number of frames to extract per second
        temperature: Temperature for GPT-4o
        save_frames: Whether to save frames to disk
        audio_transcription: Whether to transcribe audio
        resize: Resize factor for frames
        analysis_types: List of PromptType to use (defaults to GENERAL only)
        max_frames: Maximum number of frames to process

    Returns:
        Dictionary mapping shot timeframes to dictionaries of analysis results by type
    """

    if analysis_types is None:
        analysis_types = [PromptType.GENERAL]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create directory for shots
    shots_dir = os.path.join(output_dir, 'shots')
    os.makedirs(shots_dir, exist_ok=True)

    # Results dictionary
    results = {}

    # Split the video into shots
    for shot_file, timeframe in split_video(video_path, shots_dir, shot_interval, max_duration):
        shot_results = {}

        # Analyze each shot with each analysis type
        for analysis_type in analysis_types:
            print(f"\n-- Processing shot {shot_file} with {analysis_type.value} analysis --")

            analysis_result = execute_video_processing(
                shot_path=shot_file,
                analysis_type=analysis_type,
                user_prompt=USER_PROMPT,
                temperature=temperature,
                frames_per_second=frames_per_second,
                analysis_dir=output_dir,
                save_frames=save_frames,
                audio_transcription=audio_transcription,
                resize=resize,
                segment_timeframe=timeframe,
                max_frames=max_frames
            )

            shot_results[analysis_type.value] = analysis_result

        results[timeframe] = shot_results

    # Save the combined results
    combined_results_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_combined_analysis.json")
    with open(combined_results_file, 'w') as f:
        # Convert Pydantic models to dictionaries for JSON serialization
        serializable_results = {}
        for timeframe, analyses in results.items():
            serializable_results[timeframe] = {
                analysis_type: analysis.model_dump() for analysis_type, analysis in analyses.items()
            }
        json.dump(serializable_results, f, indent=4)

    return results

def create_comprehensive_analysis(video_path, output_dir='analysis_output', shot_interval=30,
                                  max_duration=None, save_frames=False, audio_transcription=True,
                                  max_frames=80):
    """
    Create a comprehensive analysis of a movie using all available analysis types

    Args:
        video_path: Path to the video file
        output_dir: Directory to save analysis results
        shot_interval: Interval in seconds to split the video
        max_duration: Maximum duration to analyze (in seconds)
        save_frames: Whether to save frames to disk
        audio_transcription: Whether to transcribe audio
        max_frames: Maximum number of frames to process per segment

    Returns:
        Dictionary of analysis results
    """

    # Use all available analysis types
    all_analysis_types = [
        PromptType.GENERAL,
        PromptType.CHARACTER,
        PromptType.VISUAL,
        PromptType.NARRATIVE,
        PromptType.CULTURAL,
        PromptType.DIRECTOR,
        PromptType.TECHNICAL,
        PromptType.SCENE
    ]

    return analyze_movie_with_multiple_perspectives(
        video_path=video_path,
        output_dir=output_dir,
        shot_interval=shot_interval,
        max_duration=max_duration,
        save_frames=save_frames,
        audio_transcription=audio_transcription,
        analysis_types=all_analysis_types,
        max_frames=max_frames
    )

if __name__ == '__main__':
    video_file = 'sample.mp4'  # Replace with the path to your video file
    output_analysis_dir = 'analysis_output'
    os.makedirs(output_analysis_dir, exist_ok=True)

    # Example 1: Process the entire video with general analysis
    print(f"\n--- Processing the entire video with general analysis: {video_file} ---")
    general_analysis = execute_video_processing(
        shot_path=video_file,
        analysis_type=PromptType.GENERAL,
        user_prompt=USER_PROMPT,
        temperature=DEFAULT_TEMPERATURE,
        frames_per_second=1,  # Lower frame rate for efficiency
        analysis_dir=output_analysis_dir,
        save_frames=False,
        audio_transcription=True,
        resize=RESIZE_OF_FRAMES
    )
    print(f"\nGeneral Analysis Summary:\n{general_analysis.model_dump_json(indent=2)}")

    # Example 2: Process the video with character-focused analysis
    print(f"\n--- Processing the video with character-focused analysis: {video_file} ---")
    character_analysis = execute_video_processing(
        shot_path=video_file,
        analysis_type=PromptType.CHARACTER,
        user_prompt=USER_PROMPT,
        temperature=DEFAULT_TEMPERATURE,
        frames_per_second=0.5,  # Even lower frame rate for character analysis
        analysis_dir=output_analysis_dir,
        save_frames=False,
        audio_transcription=True,
        resize=RESIZE_OF_FRAMES
    )
    print(f"\nCharacter Analysis Summary:\n{character_analysis.model_dump_json(indent=2)}")

    # Example 3: Process the video in shots with comprehensive analysis
    print(f"\n--- Processing video shots with comprehensive analysis: {video_file} ---")
    shot_interval = 15  # Analyze every 15 seconds
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
        print(f"  Shot {timeframe}:")
        for analysis_type, result in analyses.items():
            print(f"    - {analysis_type}: {result.scene_details.setting}")

    print("\nMovie Analyzer usage examples:")
    print("""
    # Basic usage - analyze a single video file with general film analysis
    general_analysis = execute_video_processing(
        shot_path='your_movie.mp4',
        analysis_type=PromptType.GENERAL,
        audio_transcription=True
    )
    
    # Character-focused analysis
    character_analysis = execute_video_processing(
        shot_path='your_movie.mp4', 
        analysis_type=PromptType.CHARACTER,
        audio_transcription=True
    )
    
    # Visual style analysis
    visual_analysis = execute_video_processing(
        shot_path='your_movie.mp4',
        analysis_type=PromptType.VISUAL,
        audio_transcription=True
    )
    
    # For a full cinematic breakdown of a movie in 30-second segments
    analysis = analyze_movie_with_multiple_perspectives(
        video_path='your_movie.mp4',
        shot_interval=30,
        analysis_types=[PromptType.GENERAL, PromptType.VISUAL, PromptType.CHARACTER]
    )
    
    # For a complete film studies analysis with all perspectives
    full_analysis = create_comprehensive_analysis(
        video_path='your_movie.mp4',
        shot_interval=60  # One minute segments
    )
    """)
