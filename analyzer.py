import os
import time
import json
from openai import AzureOpenAI

from config import (
    AOAI_ENDPOINT, AOAI_APIKEY, AOAI_APIVERSION, AOAI_MODEL_NAME,
    DEFAULT_TEMPERATURE, SYSTEM_PROMPTS, USER_PROMPT, DEFAULT_FRAMES_PER_SECOND, 
    RESIZE_OF_FRAMES, DEFAULT_SHOT_INTERVAL, PromptType
)
from models import VideoAnalysis, ContentAnalysis
from video_processor import process_video, split_video
from audio_processor import process_audio

# Create AOAI client for answer generation
aoai_client = AzureOpenAI(
    azure_deployment=AOAI_MODEL_NAME,
    api_version=AOAI_APIVERSION,
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_APIKEY
)

def analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature):
    """
    Analyze video frames using GPT-4o
    
    Args:
        base64frames: List of base64-encoded frames
        system_prompt: System prompt for GPT-4o
        user_prompt: User prompt for GPT-4o
        transcription: Audio transcription
        temperature: Temperature for GPT-4o
        
    Returns:
        Analysis text from GPT-4o
    """
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
                    model=AOAI_MODEL_NAME,
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
                    model=AOAI_MODEL_NAME,
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
            The following is a multi-part analysis of a video ad, broken into {num_batches} sequential segments due to technical constraints.
            Please synthesize this into a single coherent analysis, removing redundancies and creating a comprehensive overview.
            Focus on the overall messaging, audience appeal, and marketing effectiveness across the entire sequence.
            """

            consolidation_response = aoai_client.chat.completions.create(
                model=AOAI_MODEL_NAME,
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

def extract_structured_data(text_analysis, analysis_type, video_name, segment_timeframe=None, transcription=None):
    """
    Extract structured data from text analysis using GPT-4o
    
    Args:
        text_analysis: Analysis text from GPT-4o
        analysis_type: Type of analysis performed
        video_name: Name of the video file
        segment_timeframe: Timeframe of the analyzed segment
        transcription: Audio transcription
        
    Returns:
        VideoAnalysis model with structured data
    """
    print(f"Extracting structured data from text analysis for {video_name}")

    # Define the system prompt for extracting structured data
    system_prompt = f"""
    You are an expert in structured data extraction from social media ad and UGC video analysis text. 
    Your task is to carefully read the provided analysis text and extract specific information
    to create a detailed JSON object that conforms to the expected schema.
    
    The analysis type is: {analysis_type.value}
    
    VERY IMPORTANT: You must extract as much information as possible from the text and populate the JSON fields
    according to their required types. Pay careful attention to which fields require LIST values vs STRING values.
    
    If the analysis text doesn't explicitly mention something but reasonably implies it, make an educated inference
    to fill in those fields rather than leaving them empty.
    
    Here are the JSON fields you need to populate:
    
    1. video_name: Use the provided video name
    2. segment_timeframe: Use the provided timeframe
    3. analysis_type: Use the provided analysis type
    4. text_analysis: Use the full provided text analysis
    5. people: Extract any mentioned people/personalities, each with:
       - role: Person's role in the ad (influencer, actor, customer, etc.) (STRING)
       - description: Physical description (STRING)
       - actions: List of key actions (LIST OF STRINGS)
       - emotional_tone: Emotional tone conveyed (STRING)
    
    6. visuals: Details about:
       - composition: Visual composition elements (LIST OF STRINGS like ["Close-up", "Product shots"])
       - camera_movement: Camera movement described (STRING)
       - lighting: Lighting style mentioned (STRING)
       - color_scheme: Color scheme and brand alignment (STRING)
    
    7. content_details:
       - setting: Setting or context of the ad (STRING, REQUIRED)
       - platform: Apparent target platform(s) (STRING)
       - message_purpose: Purpose of the content (STRING)
       - emotional_appeal: Overall emotional appeal (STRING)
    
    8. technical_details:
       - production_quality: Production quality assessment (STRING)
       - platform_optimization: Platform-specific optimizations (STRING)
       - text_elements: Text overlays and on-screen elements (STRING)
       - notable_techniques: Notable technical aspects (LIST OF STRINGS like ["Technique1", "Technique2"])
    
    9. audience_analysis:
       - target_demographics: Target demographic groups (LIST OF STRINGS)
       - pain_points: Audience pain points addressed (LIST OF STRINGS)
       - appeal_elements: Elements designed to appeal to the audience (LIST OF STRINGS)
    
    10. brand_analysis:
        - brand_elements: Visual brand elements present (LIST OF STRINGS)
        - brand_voice: Brand voice and personality conveyed (STRING)
        - consistency: Brand consistency assessment (STRING)
        
    11. call_to_action: Call to action identified (STRING)
    12. key_messages: Key messages identified (LIST OF STRINGS)
    13. transcription: Use the provided transcription
    
    IMPORTANT: For fields that should be lists, return actual JSON arrays (not comma-separated strings).
    Example for a list field: "composition": ["Close-up", "Product shot", "Testimonial framing"]
    NOT "composition": "Close-up, Product shot, Testimonial framing"
    
    Format your response as valid JSON that can be parsed by a JSON parser.
    """

    try:
        response = aoai_client.chat.completions.create(
            model=AOAI_MODEL_NAME,
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

        # Fix potential type mismatches - ensure lists are really lists and strings are strings
        # List fields that should be corrected if they're strings
        list_fields = {
            "visuals": ["composition"],
            "technical_details": ["notable_techniques"],
            "audience_analysis": ["target_demographics", "pain_points", "appeal_elements"],
            "brand_analysis": ["brand_elements"],
            "key_messages": []  # Root level field
        }
        
        # String fields that should be corrected if they're lists
        string_fields = {
            "visuals": ["camera_movement", "lighting", "color_scheme"],
            "technical_details": ["production_quality", "platform_optimization", "text_elements"],
            "content_details": ["setting", "platform", "message_purpose", "emotional_appeal"],
            "brand_analysis": ["brand_voice", "consistency"],
            "call_to_action": None  # Root level field
        }
        
        # Fix list fields
        for section, fields in list_fields.items():
            if section in structured_data and structured_data[section]:
                for field in fields:
                    if field in structured_data[section]:
                        # If it's a string, split it into a list
                        if isinstance(structured_data[section][field], str):
                            if structured_data[section][field]:
                                # Split by commas and strip whitespace
                                structured_data[section][field] = [item.strip() for item in structured_data[section][field].split(',')]
                            else:
                                structured_data[section][field] = []
                        # If it's None, initialize as empty list
                        elif structured_data[section][field] is None:
                            structured_data[section][field] = []
        
        # Handle root level list fields
        if "key_messages" in structured_data and isinstance(structured_data["key_messages"], str):
            structured_data["key_messages"] = [item.strip() for item in structured_data["key_messages"].split(',')]
        elif "key_messages" not in structured_data or structured_data["key_messages"] is None:
            structured_data["key_messages"] = []
        
        # Fix string fields
        for section, fields in string_fields.items():
            if section in structured_data and structured_data[section] and fields:
                for field in fields:
                    if field in structured_data[section]:
                        # If it's a list, join it into a string
                        if isinstance(structured_data[section][field], list):
                            structured_data[section][field] = ", ".join(structured_data[section][field]) if structured_data[section][field] else None

        # Handle root level string fields
        if "call_to_action" in structured_data and isinstance(structured_data["call_to_action"], list):
            structured_data["call_to_action"] = ", ".join(structured_data["call_to_action"])
        
        # Ensure person actions are lists
        if "people" in structured_data and structured_data["people"]:
            for person_idx, person in enumerate(structured_data["people"]):
                if "actions" in person:
                    if isinstance(person["actions"], str):
                        # Split by commas and strip whitespace
                        structured_data["people"][person_idx]["actions"] = [item.strip() for item in person["actions"].split(',')]
                    elif person["actions"] is None:
                        structured_data["people"][person_idx]["actions"] = []
        
        # Ensure setting is not empty (required field)
        if "content_details" not in structured_data or not structured_data["content_details"]:
            structured_data["content_details"] = {"setting": "Digital content environment"}
        elif "setting" not in structured_data["content_details"] or not structured_data["content_details"]["setting"]:
            structured_data["content_details"]["setting"] = "Setting implied from analysis"

        # Try to extract setting from text if missing
        if structured_data.get("content_details", {}).get("setting") == "Unknown setting" or structured_data.get("content_details", {}).get("setting") == "Error extracting data, see text_analysis field":
            # Simple extraction of potential setting descriptions
            setting_indicators = ["takes place in", "set in", "located in", "setting is", "background shows", "filmed in", "environment is"]
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
                if "content_details" not in structured_data:
                    structured_data["content_details"] = {}
                structured_data["content_details"]["setting"] = extracted_setting

        # Ensure all required sections exist
        for section in ["visuals", "content_details", "technical_details", "audience_analysis", "brand_analysis"]:
            if section not in structured_data or not structured_data[section]:
                structured_data[section] = {}
        
        # Ensure people is a list
        if "people" not in structured_data or not structured_data["people"]:
            structured_data["people"] = []

        # Validate with Pydantic
        try:
            video_analysis = VideoAnalysis(**structured_data)
            return video_analysis
        except Exception as validation_error:
            print(f"Validation error details: {validation_error}")
            # If we still have validation errors, let's try a more aggressive approach
            
            # Ensure all list fields exist and are proper lists
            for section, fields in list_fields.items():
                if section not in structured_data:
                    structured_data[section] = {}
                for field in fields:
                    if field and section in structured_data:  # Skip empty field names (for root level fields)
                        structured_data[section][field] = structured_data.get(section, {}).get(field, [])
                        if not isinstance(structured_data[section][field], list):
                            structured_data[section][field] = []
            
            # Make sure root level list fields exist
            if "key_messages" not in structured_data or not isinstance(structured_data["key_messages"], list):
                structured_data["key_messages"] = []
            
            # Try again with fixed data
            video_analysis = VideoAnalysis(**structured_data)
            return video_analysis

    except Exception as ex:
        print(f'ERROR extracting structured data: {ex}')
        # Create a minimal valid object as fallback
        return VideoAnalysis(
            video_name=video_name,
            analysis_type=analysis_type,
            text_analysis=text_analysis,
            content_details=ContentAnalysis(setting="Error extracting data, see text_analysis field"),
            transcription=transcription
        )

def execute_video_processing(shot_path, analysis_type, user_prompt=USER_PROMPT, temperature=DEFAULT_TEMPERATURE,
                             frames_per_second=DEFAULT_FRAMES_PER_SECOND, analysis_dir='analysis_output',
                             save_frames=False, audio_transcription=False, resize=RESIZE_OF_FRAMES,
                             segment_timeframe=None, max_frames=80):
    """
    Process a video file with the specified analysis type
    
    Args:
        shot_path: Path to the video file
        analysis_type: Type of analysis to perform
        user_prompt: User prompt for GPT-4o
        temperature: Temperature for GPT-4o
        frames_per_second: Number of frames to extract per second
        analysis_dir: Directory to save analysis results
        save_frames: Whether to save frames to disk
        audio_transcription: Whether to transcribe audio
        resize: Factor to resize frames by
        segment_timeframe: Timeframe of the analyzed segment
        max_frames: Maximum number of frames to process
        
    Returns:
        VideoAnalysis model with analysis results
    """
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
    print(f"Analyzing frames with {AOAI_MODEL_NAME} using {analysis_type.value} prompt")
    print(f"Processing {len(base64frames)} frames in batches")
    start_time = time.time()
    text_analysis = analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature)
    end_time = time.time()
    print(f'\t>>>> Analysis with {AOAI_MODEL_NAME} took {(end_time - start_time):.3f} seconds <<<<')

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

    return structured_data

def analyze_video_with_multiple_perspectives(video_path, output_dir='analysis_output', shot_interval=DEFAULT_SHOT_INTERVAL,
                                            max_duration=None, frames_per_second=DEFAULT_FRAMES_PER_SECOND,
                                            temperature=DEFAULT_TEMPERATURE, save_frames=False,
                                            audio_transcription=True, resize=RESIZE_OF_FRAMES,
                                            analysis_types=None, max_frames=80):
    """
    Analyze a video from multiple perspectives by using different prompt types

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

    # Generate a consolidated analysis
    consolidated_analysis = create_consolidated_analysis(video_path, results, output_dir)
    
    return results

def create_consolidated_analysis(video_path, results, output_dir):
    """
    Create a consolidated analysis of all shots by analyzing the individual shot analyses.
    
    Args:
        video_path: Path to the video file
        results: Dictionary of analysis results by timeframe and analysis type
        output_dir: Directory to save the consolidated analysis
        
    Returns:
        Consolidated analysis as a JSON object
    """
    print("\n--- Creating consolidated analysis of all shots ---")
    
    # Prepare data for the consolidation prompt
    consolidated_data = {
        "video_name": os.path.basename(video_path),
        "total_shots": len(results),
        "shot_analyses": []
    }
    
    # Organize the data by timeframe for the prompt
    for timeframe, analyses in results.items():
        shot_data = {
            "timeframe": timeframe,
            "analyses": {}
        }
        
        # Add each analysis type for this shot
        for analysis_type, analysis in analyses.items():
            # Extract key information for the consolidated analysis
            shot_data["analyses"][analysis_type] = {
                "setting": analysis.content_details.setting if hasattr(analysis, 'content_details') and analysis.content_details else "Unknown",
                "people": [{"role": person.role, "actions": person.actions} for person in analysis.people] if hasattr(analysis, 'people') and analysis.people else [],
                "emotional_appeal": analysis.content_details.emotional_appeal if hasattr(analysis, 'content_details') and analysis.content_details and analysis.content_details.emotional_appeal else "Unknown",
                "message_purpose": analysis.content_details.message_purpose if hasattr(analysis, 'content_details') and analysis.content_details and analysis.content_details.message_purpose else "Unknown",
                "key_messages": analysis.key_messages if hasattr(analysis, 'key_messages') and analysis.key_messages else []
            }
        
        consolidated_data["shot_analyses"].append(shot_data)
    
    # Sort by timeframe to ensure chronological order
    consolidated_data["shot_analyses"].sort(key=lambda x: x["timeframe"])
    
    # Create the system prompt for the consolidated analysis
    system_prompt = """
    You are an expert social media ad and UGC content analyst specializing in marketing effectiveness.
    Your task is to create a comprehensive analysis of a video ad based on the analyses of individual segments.
    
    Analyze the progression of the ad and create a unified analysis that:
    
    1. Summarizes the overall messaging and value proposition
    2. Identifies primary audience targeting and appeal factors
    3. Analyzes the visual branding and creative approach
    4. Highlights key conversion elements and call-to-action effectiveness
    5. Provides an overall assessment of marketing effectiveness
    
    For each time segment, maintain the timeline context (the timeframe) in your analysis. This is crucial
    for understanding the progression of the ad.
    
    Your consolidated analysis should be more than a summary of individual segments - it should provide
    insights into how the segments work together to create an effective marketing message.
    """
    
    # Create the user prompt with the consolidated data
    user_prompt = f"""
    Below is information about a video ad that has been analyzed in {len(results)} separate segments.
    Create a comprehensive consolidated analysis of the entire ad, paying attention to:
    
    - Messaging progression and key points
    - Audience targeting and emotional appeals
    - Visual branding and creative execution
    - Call-to-action effectiveness
    - Overall marketing impact
    
    Include timestamps/timeframes for significant elements to maintain the chronological structure.
    
    Video information:
    {json.dumps(consolidated_data, indent=2)}
    """
    
    try:
        # Generate the consolidated analysis
        print("Generating consolidated analysis...")
        response = aoai_client.chat.completions.create(
            model=AOAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=4096
        )
        
        consolidated_text = response.choices[0].message.content
        
        # Save the consolidated analysis
        consolidated_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_consolidated_analysis.txt")
        with open(consolidated_file, 'w') as f:
            f.write(consolidated_text)
        print(f"Consolidated analysis saved as: {consolidated_file}")
        
        # Also save as JSON for structured access
        consolidated_json_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_final_analysis.json")
        
        # Extract structured data from the consolidated analysis
        system_prompt_structured = """
        You are an expert in extracting structured data from marketing content analysis.
        Extract the key information from the consolidated ad analysis into a structured format.
        
        Include:
        1. overall_messaging: A summary of the key messages and value proposition
        2. target_audience: Information about the target demographic and psychographic profile
        3. visual_branding: Description of visual elements and brand presentation
        4. key_moments: List of significant moments with their timeframes
        5. marketing_tactics: Marketing tactics and persuasion techniques used
        6. conversion_elements: How the ad attempts to drive viewer action
        7. effectiveness_score: A subjective rating from 1-10 of overall effectiveness
        """
        
        structured_response = aoai_client.chat.completions.create(
            model=AOAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt_structured},
                {"role": "user", "content": consolidated_text}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=4096
        )
        
        consolidated_structured = json.loads(structured_response.choices[0].message.content)
        consolidated_structured["full_analysis"] = consolidated_text
        consolidated_structured["video_name"] = os.path.basename(video_path)
        
        with open(consolidated_json_file, 'w') as f:
            json.dump(consolidated_structured, f, indent=4)
        print(f"Structured consolidated analysis saved as: {consolidated_json_file}")
        
        return consolidated_structured
        
    except Exception as ex:
        print(f"ERROR creating consolidated analysis: {ex}")
        return {"error": str(ex), "video_name": os.path.basename(video_path)}

def create_comprehensive_analysis(video_path, output_dir='analysis_output', shot_interval=DEFAULT_SHOT_INTERVAL,
                                  max_duration=None, save_frames=False, audio_transcription=True,
                                  max_frames=80):
    """
    Create a comprehensive analysis of a video ad using all available analysis types

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
        PromptType.AUDIENCE,
        PromptType.VISUAL,
        PromptType.MESSAGING,
        PromptType.BRANDING,
        PromptType.ENGAGEMENT,
        PromptType.TECHNICAL,
        PromptType.CONVERSION
    ]

    return analyze_video_with_multiple_perspectives(
        video_path=video_path,
        output_dir=output_dir,
        shot_interval=shot_interval,
        max_duration=max_duration,
        save_frames=save_frames,
        audio_transcription=audio_transcription,
        analysis_types=all_analysis_types,
        max_frames=max_frames
    ) 