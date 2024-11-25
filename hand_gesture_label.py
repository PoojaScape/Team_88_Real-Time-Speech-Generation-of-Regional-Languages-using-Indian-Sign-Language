import cv2
import mediapipe as mp
import os
import pandas as pd
import itertools
import copy

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=2)

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    temp_landmark_list = list(map(lambda n: n / max_value if max_value else 0, temp_landmark_list))
    return temp_landmark_list

def process_frame(image, window_size=(640, 480)):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
            processed_landmarks = pre_process_landmark(landmark_list)
            keypoints.extend(processed_landmarks)
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        while len(keypoints) < 84:
            keypoints.extend([-4, -4])
        keypoints = keypoints[:84]
        resized_frame = cv2.resize(image, window_size)
        return resized_frame, keypoints
    else:
        return None, [-4] * 84

def extract_and_save_frames(video_path, label, video_id, output_dir, csv_file):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    data = []
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame or end of video {video_path}.")
            break
        processed_frame, keypoints = process_frame(frame)
        if processed_frame is not None and keypoints is not None:
            cv2.imshow('Hand Landmarks', processed_frame)
            data.append([label, video_id, frame_count] + keypoints)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    if data:
        df = pd.DataFrame(data, columns=['Gesture', 'Video_ID', 'Frame'] + [f'Keypoint_{i+1}' for i in range(84)])
        df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)

def main():
    base_dir = 'OtherPhrases'
    output_dir = 'frames_with_landmarks'
    csv_file = 'Test.csv'
    os.makedirs(output_dir, exist_ok=True)
    labels = os.listdir(base_dir)
    
    start_video_id = 0
    end_video_id = 1
    for label_no in range(len(labels)):
        for video_id in range(start_video_id, end_video_id):
            video_file = f'{video_id}.mp4'
            video_path = os.path.join(base_dir, labels[label_no], video_file)
            print(f"Processing video {video_path} with label {labels[label_no]} and video ID {video_id}...")
            extract_and_save_frames(video_path, labels[label_no], video_id, output_dir, csv_file)

if __name__ == "__main__":
    main()
