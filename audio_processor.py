import os
import logging
from moviepy.editor import VideoFileClip
from config import get_whisper_client, WHISPER_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_audio(video_path):
    """
    Extract audio from a video file and transcribe it using Whisper.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        str: Transcription text as a string
    """
    logger.info(f"Starting audio transcription for {video_path}")
    transcription_text = ''
    
    try:
        # Create audio path
        base_video_path, _ = os.path.splitext(video_path)
        audio_path = f"{base_video_path}.mp3"
        
        # Check if audio file already exists
        if not os.path.exists(audio_path):
            logger.info(f"Extracting audio to {audio_path}")
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(audio_path, bitrate="32k", logger=None)
            clip.audio.close()
            clip.close()
        else:
            logger.info(f"Using existing audio file: {audio_path}")

        # Get whisper client from factory function
        whisper_client = get_whisper_client()
            
        # Transcribe the audio
        logger.info(f"Transcribing audio from {audio_path}")
        
        try:
            with open(audio_path, "rb") as audio_file:
                transcription = whisper_client.audio.transcriptions.create(
                    model=WHISPER_MODEL_NAME,
                    file=audio_file,
                )
                transcription_text = transcription.text
                logger.info(f"Transcript: {transcription_text}")
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_path}")
            raise
        except Exception as ex:
            logger.error(f'Error during transcription: {ex}')
            raise
            
    except Exception as ex:
        logger.error(f'Error in audio processing: {ex}')
        transcription_text = ''

    return transcription_text 