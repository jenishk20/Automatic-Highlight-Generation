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
    return sum(1 for file in os.listdir(folder_path) if file.endswith(".mp4"))

def extract_events_with_context(data, context_time=5000, overlap_threshold=3000):
    
    events = []

    for idx, annotation in enumerate(data["annotations"]):
        event_label = annotation["label"]
        event_position = int(annotation["position"])
        game_time = annotation["gameTime"]
        half = game_time.split(" - ")[0]
        start_time = max(0, event_position - context_time)
        end_time = event_position + context_time
        label_rank = RANKINGS.get(event_label, 5)

        is_overlapping = False
        for existing_event in events:
            if existing_event["half"] == half:
                if abs(existing_event["event_position"] - event_position) <= overlap_threshold:
                    existing_rank = RANKINGS.get(existing_event["label"], 5)
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
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    start_time_sec = start_time / 1000.0
    end_time_sec = end_time / 1000.0

    video.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)

    frames = []
    frame_count = 0
    success = True

    total_frames_to_extract = int((end_time_sec - start_time_sec) * frame_rate)

    while success and frame_count < total_frames_to_extract:
        success, frame = video.read()

        if not success:
            break

        frames.append(frame)

        frame_count += 1

    video.release()
    return frames


def create_video_from_frames(frames, output_video_path, frame_rate=30):
    if not frames:
        print(f"No frames to create video at {output_video_path}.")
        return

    height, width, layers = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print(f"Video created at {output_video_path}")


def extract_event_frames(data, video_path, output_dir, context_time=5000):
    events = extract_events_with_context(data, context_time)
    print(video_path)

    for event in events:
        event_label = event["label"]
        event_position = event["event_position"]
        start_time = event["start_time"]
        end_time = event["end_time"]
        half = event["gameTime"][:1]
        unique_index = event["index"]
        
        path = os.path.join(video_path, f"{half}_720p.mkv")
        if not os.path.exists(path):
            print(f"Error: Video file {path} does not exist.")
            continue

        label_rank = RANKINGS.get(event_label, 5)  # Default rank to 5 if label is not in RANKINGS
        folder_name = event_label if label_rank <= 5 else "nothing"

        # Create label-specific folder if not exists
        label_dir = os.path.join(output_dir, folder_name)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        
        video_count = count_videos_in_folder(label_dir)
        if video_count >= 100:
            print(f"Skipping frame extraction for {folder_name}, as it already contains {video_count} videos.")
        else:
        # Extract frames
            frames = extract_frames_in_timeframe(path, start_time, end_time)

            # Create a video for the event
            if frames:
                output_video_path = os.path.join(label_dir, f"{folder_name}_half_{half}_at_{event_position}_index_{unique_index}.mp4")
                create_video_from_frames(frames, output_video_path)

    return events


def process_league_videos(root_path):
    for league in os.listdir(root_path):
        league_path = os.path.join(root_path, league)
        if not os.path.isdir(league_path):
            continue

        for season in os.listdir(league_path):
            season_path = os.path.join(league_path, season)
            if not os.path.isdir(season_path):
                continue

            for match in os.listdir(season_path):
                match_path = os.path.join(season_path, match)
                if not os.path.isdir(match_path):
                    continue

                # Load the JSON file
                label_file_path = os.path.join(match_path, "Labels-v2.json")
                if not os.path.exists(label_file_path):
                    print(f"No label file found in {match_path}. Skipping...")
                    continue

                with open(label_file_path) as f:
                    data = json.load(f)

                # Define output directory
                output_dir = os.path.join(root_path, "extracted")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Extract frames and create videos
                print(f"Processing match: {match}")
                extract_event_frames(data, match_path, output_dir)


root_path = "../data/"
process_league_videos(root_path)