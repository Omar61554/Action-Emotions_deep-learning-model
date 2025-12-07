import os
import cv2
import numpy as np
import random

# ----------------------------------------------------------
# Choose a set of augmentations once per video
# ----------------------------------------------------------
def choose_augmentation_plan():
    return {
        "flip": random.random() < 0.5,
        "brightness": random.random() < 0.5,
        "brightness_value": random.randint(-40, 40),

        "rotation": random.random() < 0.3,
        "rotation_angle": random.uniform(-15, 15),

        "blur": random.random() < 0.3,
        "blur_kernel": random.choice([3, 5]),

        "noise": random.random() < 0.3,
        "noise_std": random.uniform(5, 15)
    }


# ----------------------------------------------------------
# Apply SAME augmentation parameters to ALL frames
# ----------------------------------------------------------
def apply_augmentations(frame, plan):

    if plan["flip"]:
        frame = cv2.flip(frame, 1)

    if plan["brightness"]:
        value = plan["brightness_value"]
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.int16)
        h, s, v = cv2.split(hsv)
        v = np.clip(v + value, 0, 255)
        hsv = np.stack([h, s, v], axis=-1).astype(np.uint8)
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if plan["rotation"]:
        angle = plan["rotation_angle"]
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if plan["blur"]:
        k = plan["blur_kernel"]
        frame = cv2.GaussianBlur(frame, (k, k), 0)

    if plan["noise"]:
        std = plan["noise_std"]
        noise = np.random.normal(0, std, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame


# ----------------------------------------------------------
# Extract frames from ONE video
# ----------------------------------------------------------
def extract_frames(video_path, output_folder, num_frames=32, augment=True):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 1:
        print("Error: cannot read video:", video_path)
        return

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    plan = choose_augmentation_plan() if augment else None
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    print("Augmentation plan:", plan)

    idx = 0  
    for frame_pos in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_pos))
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))

        if augment:
            frame = apply_augmentations(frame, plan)

        save_path = os.path.join(output_folder, f"frame_{idx:03d}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        idx += 1

    cap.release()
    print(f"Saved {idx} frames in {output_folder}")


# ----------------------------------------------------------
# Process ONE action folder (Basketball, Diving, etc.)
# ----------------------------------------------------------
def process_action_folder(input_action_folder, output_action_folder, num_videos=40, start_index=0):

    # Ensure output folder for this action exists
    os.makedirs(output_action_folder, exist_ok=True)

    # Get all video files
    all_videos = sorted([
        f for f in os.listdir(input_action_folder)
        if f.lower().endswith(('.avi', '.mp4', '.mov'))
    ])

    # Limit to first N videos
    #selected_videos = all_videos[:num_videos]

    selected_videos = all_videos[start_index : start_index + num_videos]


    print(f"\n=== Processing action folder: {input_action_folder} ===")
    print(f"Found {len(all_videos)} videos, taking first {len(selected_videos)}\n")

    for video_name in selected_videos:

        video_path = os.path.join(input_action_folder, video_name)

        # Folder named after the video inside output
        video_output_folder = os.path.join(
            output_action_folder,
            os.path.splitext(video_name)[0]
        )

        extract_frames(video_path, video_output_folder, num_frames=32, augment=True)

    print(f"\nCompleted action: {os.path.basename(input_action_folder)}")
    print(f"Frames saved in: {output_action_folder}\n")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":

    input_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\UCF-101\CliffDiving"
    output_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\Trials codes\extracted_frames\CliffDiving"

    process_action_folder(input_action_folder, output_action_folder, num_videos=138, start_index=0)

    input_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\UCF-101\WritingOnBoard"
    output_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\Trials codes\extracted_frames\WritingOnBoard"

    process_action_folder(input_action_folder, output_action_folder, num_videos=152, start_index=0)

    input_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\UCF-101\TaiChi"
    output_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\Trials codes\extracted_frames\TaiChi"

    process_action_folder(input_action_folder, output_action_folder, num_videos=100, start_index=0)

    input_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\UCF-101\SalsaSpin"
    output_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\Trials codes\extracted_frames\SalsaSpin"

    process_action_folder(input_action_folder, output_action_folder, num_videos=133, start_index=0)

    input_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\UCF-101\MilitaryParade"
    output_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\Trials codes\extracted_frames\MilitaryParade"

    process_action_folder(input_action_folder, output_action_folder, num_videos=125, start_index=0)

    input_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\UCF-101\Haircut"
    output_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\Trials codes\extracted_frames\Haircut"

    process_action_folder(input_action_folder, output_action_folder, num_videos=130, start_index=0)

    input_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\UCF-101\Diving"
    output_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\Trials codes\extracted_frames\Diving"

    process_action_folder(input_action_folder, output_action_folder, num_videos=150,start_index=0)

    input_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\UCF-101\Basketball"
    output_action_folder = r"D:\Eng ASU\Fall 2025 Senior 2 Mechatronics\Machine Vision\Project\Trials codes\extracted_frames\Basketball"

    process_action_folder(input_action_folder, output_action_folder, num_videos=134, start_index=0)
#change the input_action_folder and output_action_folder variables to process other action folders
#selected actions are: Basketball - Diving - Haircut - MilitaryParade - SalsaSpin - TaiChi - WritingOnBoard - CliffDiving
