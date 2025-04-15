import os
import json
import cv2


# Rankings of event labels
RANKINGS = {
    "Kick-off": 1,
    "Goal": 2,
    "Shots on target": 3,
    "Corner": 4,
    "Foul": 5,
}

def count_videos_in_folder(folder_path):
    """
    Counts the number of videos in the folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        int: Number of MP4 videos in the folder.
    """
    try:
        # Count files with .mp4 extension in the folder
        return sum(1 for file in os.listdir(folder_path) if file.lower().endswith(".mp4"))
    except FileNotFoundError:
        print(f"Error: Folder {folder_path} does not exist.")
        return 0


def extract_events_with_context(data, context_time=5000, overlap_threshold=3000):
    """
    Extracts events with context time and resolves overlapping events.

    Args:
        data (dict): JSON data containing event annotations.
        context_time (int): Time (in ms) before and after the event to include.
        overlap_threshold (int): Time (in ms) to determine overlapping events.

    Returns:
        list: List of events with context information.
    """
    events = []

    # Iterate through each annotation in the data
    for idx, annotation in enumerate(data["annotations"]):
        event_label = annotation["label"]
        event_position = int(annotation["position"])
        game_time = annotation["gameTime"]
        half = game_time.split(" - ")[0]
        start_time = max(0, event_position - context_time)
        end_time = event_position + context_time
        label_rank = RANKINGS.get(event_label, 5)

        is_overlapping = False
        # Check for overlapping events
        for existing_event in events:
            if existing_event["half"] == half:
                if abs(existing_event["event_position"] - event_position) <= overlap_threshold:
                    existing_rank = RANKINGS.get(existing_event["label"], 5)
                    # Update event if the new one has a higher rank
                    if label_rank < existing_rank:
                        existing_event.update({
                            "label": event_label,
                            "event_position": event_position,
                            "start_time": start_time,
                            "end_time": end_time,
                            "gameTime": game_time,
                            "index": idx
                        })
                    is_overlapping = True
                    break

        # Add non-overlapping events to the list
        if not is_overlapping:
            events.append({
                "label": event_label,
                "event_position": event_position,
                "start_time": start_time,
                "end_time": end_time,
                "gameTime": game_time,
                "half": half,
                "index": idx
            })

    return events


def extract_frames_in_timeframe(video_path, start_time, end_time, frame_rate=30):
    """
    Extracts frames from a video within a specified time frame.

    Args:
        video_path (str): Path to the video file.
        start_time (int): Start time in ms.
        end_time (int): End time in ms.
        frame_rate (int): Frame rate for extraction.

    Returns:
        list: List of extracted frames.
    """
    video = cv2.VideoCapture(video_path)

    # Check if the video file can be opened
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    start_time_sec = start_time / 1000.0
    end_time_sec = end_time / 1000.0

    # Set the video position to the start time
    video.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

    frames = []
    frame_count = 0
    success = True

    # Calculate the total number of frames to extract
    total_frames_to_extract = int((end_time_sec - start_time_sec) * frame_rate)

    # Read frames from the video
    while success and frame_count < total_frames_to_extract:
        success, frame = video.read()

        if not success:
            break

        frames.append(frame)
        frame_count += 1

    video.release()
    return frames


def create_video_from_frames(frames, output_video_path, frame_rate=30):
    """
    Creates a video from a list of frames.

    Args:
        frames (list): List of frames to include in the video.
        output_video_path (str): Path to save the output video.
        frame_rate (int): Frame rate for the output video.
    """
    # Check if there are frames to create a video
    if not frames:
        print(f"No frames to create video at {output_video_path}.")
        return

    # Get the dimensions of the frames
    height, width, layers = frames[0].shape

    # Define the video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video created at {output_video_path}")


def extract_event_frames(data, video_path, output_dir, context_time=5000):
    """
    Extracts event frames from a video based on event annotations and saves them as separate videos.

    Args:
        data (dict): The JSON data containing event annotations.
        video_path (str): The path to the video file.
        output_dir (str): The directory where extracted videos will be saved.
        context_time (int): The time (in ms) before and after the event to include in the video.
    """
    # Extract events with context from the JSON data
    events = extract_events_with_context(data, context_time)
    print(video_path)

    for event in events:
        event_label = event["label"]  # Event label (e.g., "Goal", "Foul")
        event_position = event["event_position"]  # Event position in milliseconds
        start_time = event["start_time"]  # Start time of the event context
        end_time = event["end_time"]  # End time of the event context
        half = event["gameTime"][:1]  # Extract the half (e.g., "1", "2") from gameTime
        unique_index = event["index"]  # Unique index of the event in the annotations
        
        # Path to the video file for the specific half
        path = os.path.join(video_path, f"{half}_720p.mkv")
        if not os.path.exists(path):
            print(f"Error: Video file {path} does not exist.")
            continue

        label_rank = RANKINGS.get(event_label, 5)  # Default rank to 5 if label is not in RANKINGS
        folder_name = event_label if label_rank <= 5 else "nothing"

        # Create label-specific folder if it doesn't exist
        label_dir = os.path.join(output_dir, folder_name)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Check if the folder already contains 100 videos
        video_count = count_videos_in_folder(label_dir)
        if video_count >= 100:
            print(f"Skipping frame extraction for {folder_name}, as it already contains {video_count} videos.")
        else:
            # Extract frames for the event
            frames = extract_frames_in_timeframe(path, start_time, end_time)

            # Create a video for the event if frames were successfully extracted
            if frames:
                output_video_path = os.path.join(label_dir, f"{folder_name}_half_{half}_at_{event_position}_index_{unique_index}.mp4")
                create_video_from_frames(frames, output_video_path)

    return events


def process_league_videos(root_path):
    """
    Processes league videos by iterating through a directory of leagues, seasons, and matches,
    extracting event frames and creating videos based on event labels stored in JSON files.

    Args:
        root_path (str): The root directory containing league folders. Each league folder contains season folders,
                         and each season folder contains match folders with event label JSON files.
    """
    # Iterate through each league in the root directory
    for league in os.listdir(root_path):
        league_path = os.path.join(root_path, league)
        if not os.path.isdir(league_path):
            continue  # Skip if not a directory

        # Iterate through each season in the league directory
        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            if not os.path.isdir(season_path):
                continue  # Skip if not a directory

            # Iterate through each match in the season directory
            for match in os.listdir(season_path):
                match_path = os.path.join(season_path, match)
                if not os.path.isdir(match_path):
                    continue  # Skip if not a directory

                # Load the JSON file containing event labels
                label_file_path = os.path.join(match_path, "Labels-v2.json")
                if not os.path.exists(label_file_path):
                    print(f"No label file found in {match_path}. Skipping...")
                    continue  # Skip if the label file does not exist

                with open(label_file_path) as f:
                    data = json.load(f)  # Load the JSON data

                # Define the output directory for extracted videos
                output_dir = os.path.join(root_path, "extracted")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)  # Create the output directory if it doesn't exist

                # Extract frames and create videos for the match
                print(f"Processing match: {match}")
                extract_event_frames(data, match_path, output_dir)


root_path = "../data/"
process_league_videos(root_path)