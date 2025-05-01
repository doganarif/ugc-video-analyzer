import os
from moviepy.editor import VideoFileClip
from openai import AzureOpenAI
from config import WHISPER_ENDPOINT, WHISPER_APIKEY, WHISPER_APIVERSION, WHISPER_MODEL_NAME

# Create AOAI client for whisper
whisper_client = AzureOpenAI(
    api_version=WHISPER_APIVERSION,
    azure_endpoint=WHISPER_ENDPOINT,
    api_key=WHISPER_APIKEY
)

def process_audio(video_path):
    """
    Extract audio from a video file and transcribe it using Whisper.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Transcription text as a string
    """
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
            model=WHISPER_MODEL_NAME,
            file=open(audio_path, "rb"),
        )
        transcription_text = transcription.text
        print("Transcript: ", transcription_text + "\n\n")
    except Exception as ex:
        print(f'ERROR Transcript: {ex}')
        transcription_text = ''

    return transcription_text 