import cv2
import os
import base64
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from config import DEFAULT_FRAMES_PER_SECOND, RESIZE_OF_FRAMES

def process_video(video_path, frames_per_second=DEFAULT_FRAMES_PER_SECOND, resize=RESIZE_OF_FRAMES, output_dir='',
                  max_frames=100):
    """
    Process a video file to extract frames at a specified rate.
    
    Args:
        video_path: Path to the video file
        frames_per_second: Number of frames to extract per second
        resize: Factor to resize frames by (0 for no resize)
        output_dir: Directory to save extracted frames (if not empty)
        max_frames: Maximum number of frames to extract (0 for unlimited)
        
    Returns:
        List of base64-encoded frames
    """
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

def split_video(video_path, output_dir, shot_interval, max_duration=None):
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