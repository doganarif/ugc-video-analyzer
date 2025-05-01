import os
import time
import json
import logging
from config import (
    get_aoai_client, MAX_IMAGES_PER_REQUEST,
    DEFAULT_TEMPERATURE, SYSTEM_PROMPTS, USER_PROMPT, DEFAULT_FRAMES_PER_SECOND, 
    RESIZE_OF_FRAMES, DEFAULT_SHOT_INTERVAL, PromptType, AOAI_MODEL_NAME
)
from models import VideoAnalysis, ContentAnalysis
from video_processor import process_video, split_video
from audio_processor import process_audio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_video(base64frames, system_prompt, user_prompt, transcription='', temperature=0.5):
    """
    Analyzes the frames from a video to provide a detailed description and analysis.
    """
    logger.info(f"Starting video analysis with temperature {temperature}")
    
    aoai_client = get_aoai_client()
    
    # Process frames in batches to stay within token limits
    max_images_per_request = MAX_IMAGES_PER_REQUEST
    
    if max_images_per_request >= len(base64frames):
        # If we can process all frames at once
        logger.info(f"Processing all {len(base64frames)} frames at once")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt}
                ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in base64frames]
            }
        ]
        
        # Add transcription if available
        if transcription:
            messages[1]["content"].append({"type": "text", "text": f"\nTranscription: {transcription}"})
        
        try:
            response = aoai_client.chat.completions.create(
                model=AOAI_MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in API call: {e}")
            # Try again with a smaller batch
            logger.info(f"Retrying with a smaller batch of frames")
            max_images_per_request = len(base64frames) // 2
    
    # If we need to process in batches
    logger.info(f"Processing {len(base64frames)} frames in {len(base64frames) // max_images_per_request + 1} batches")
    
    # Create batches of frames
    batches = [base64frames[i:i + max_images_per_request] for i in range(0, len(base64frames), max_images_per_request)]
    logger.info(f"Created {len(batches)} batches")
    
    combined_analysis = ""
    
    for i, batch in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)} with {len(batch)} frames")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{user_prompt} (Batch {i+1} of {len(batches)})"}
                ] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in batch]
            }
        ]
        
        # Add transcription if available and it's the first batch
        if transcription and i == 0:
            messages[1]["content"].append({"type": "text", "text": f"\nTranscription: {transcription}"})
        
        # Add previous analysis if not the first batch
        if i > 0:
            messages.append({
                "role": "assistant",
                "content": combined_analysis
            })
            messages.append({
                "role": "user",
                "content": f"Continue your analysis with the next set of frames. Consider what you've already analyzed and focus on new observations and developments."
            })
        
        try:
            response = aoai_client.chat.completions.create(
                model=AOAI_MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )
            
            batch_analysis = response.choices[0].message.content
            
            if i == 0:
                combined_analysis = batch_analysis
            else:
                combined_analysis += "\n\n" + batch_analysis
        except Exception as e:
            logger.error(f"Error in API call for batch {i+1}: {e}")
            return f"Error in analysis: {str(e)}"
    
    # Final consolidation if we processed in multiple batches
    if len(batches) > 1:
        try:
            logger.info("Creating final consolidated analysis")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": combined_analysis},
                {"role": "user", "content": "Please provide a final, consolidated analysis that integrates all of your observations into a cohesive whole. Remove any redundancies and organize your insights into a clear, well-structured analysis."}
            ]
            
            response = aoai_client.chat.completions.create(
                model=AOAI_MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in final consolidation: {e}")
    
    return combined_analysis

def extract_structured_data(text_analysis, analysis_type, video_name, segment_timeframe=None, transcription=None):
    """
    Extract structured data from text analysis using GPT-4o
    
    Args:
        text_analysis (str): Analysis text from GPT-4o
        analysis_type (PromptType): Type of analysis performed
        video_name (str): Name of the video file
        segment_timeframe (str, optional): Timeframe of the analyzed segment
        transcription (str, optional): Audio transcription
        
    Returns:
        VideoAnalysis: Model with structured data
    """
    logger.info(f"Extracting structured data from text analysis for {video_name}")

    # Get AOAI client
    aoai_client = get_aoai_client()
    
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
            logger.error(f"Validation error details: {validation_error}")
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
        logger.error(f'ERROR extracting structured data: {ex}')
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
        shot_path (str): Path to the video file
        analysis_type (PromptType): Type of analysis to perform
        user_prompt (str, optional): User prompt for GPT-4o
        temperature (float, optional): Temperature for GPT-4o
        frames_per_second (float, optional): Number of frames to extract per second
        analysis_dir (str, optional): Directory to save analysis results
        save_frames (bool, optional): Whether to save frames to disk
        audio_transcription (bool, optional): Whether to transcribe audio
        resize (int, optional): Factor to resize frames by
        segment_timeframe (str, optional): Timeframe of the analyzed segment
        max_frames (int, optional): Maximum number of frames to process
        
    Returns:
        VideoAnalysis: Model with analysis results
    """
    logger.info(f"Starting video processing for shot {shot_path} with analysis_type={analysis_type.value}")

    # Ensure analysis directory exists
    os.makedirs(analysis_dir, exist_ok=True)

    # Get the appropriate system prompt
    system_prompt = SYSTEM_PROMPTS[analysis_type]

    # Extract frames at the specified rate
    logger.info(f"Extracting frames from {shot_path}")
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
    logger.info(f'\t>>>> Frames extraction took {(end_time - start_time):.3f} seconds <<<<')

    # Extract the transcription of the audio
    transcription = ''
    if audio_transcription:
        logger.info(f"Transcribing audio from {shot_path}")
        start_time = time.time()
        transcription = process_audio(shot_path)
        end_time = time.time()
        logger.info(f'Transcription: [{transcription}]')
        logger.info(f'\t>>>> Audio transcription took {(end_time - start_time):.3f} seconds <<<<')
    else:
        logger.info(f"Skipping audio transcription")

    # Get AOAI client
    aoai_client = get_aoai_client()
    
    # Analyze the video frames and the audio transcription with GPT-4o
    logger.info(f"Analyzing frames with {AOAI_MODEL_NAME} using {analysis_type.value} prompt")
    logger.info(f"Processing {len(base64frames)} frames in batches")
    start_time = time.time()
    text_analysis = analyze_video(base64frames, system_prompt, user_prompt, transcription, temperature)
    end_time = time.time()
    logger.info(f'\t>>>> Analysis with {AOAI_MODEL_NAME} took {(end_time - start_time):.3f} seconds <<<<')

    # Extract structured data from the text analysis
    logger.info(f"Extracting structured data from text analysis")
    start_time = time.time()
    structured_data = extract_structured_data(text_analysis, analysis_type,
                                              os.path.basename(shot_path),
                                              segment_timeframe, transcription)
    end_time = time.time()
    logger.info(f'\t>>>> Structured data extraction took {(end_time - start_time):.3f} seconds <<<<')

    # Save the analysis to a JSON file in the analysis directory
    analysis_filename = os.path.join(analysis_dir,
                                     f"{os.path.splitext(os.path.basename(shot_path))[0]}_{analysis_type.value}_analysis.json")
    with open(analysis_filename, 'w') as json_file:
        json.dump(structured_data.model_dump(), json_file, indent=4)
    logger.info(f"Analysis saved as: {analysis_filename}")

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
            logger.info(f"\n-- Processing shot {shot_file} with {analysis_type.value} analysis --")

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
    create_consolidated_analysis(video_path, results, output_dir)
    
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
    logger.info("\n--- Creating consolidated analysis of all shots ---")
    
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
        logger.info("Generating consolidated analysis...")
        response = get_aoai_client().chat.completions.create(
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
        logger.info(f"Consolidated analysis saved as: {consolidated_file}")
        
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
        
        structured_response = get_aoai_client().chat.completions.create(
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
        logger.info(f"Structured consolidated analysis saved as: {consolidated_json_file}")
        
        return consolidated_structured
        
    except Exception as ex:
        logger.error(f"ERROR creating consolidated analysis: {ex}")
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