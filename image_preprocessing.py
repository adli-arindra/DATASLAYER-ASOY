import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import shutil

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_landmarks(image_path:str = "") -> None:
    with mp_pose.Pose(min_detection_confidence=0, min_tracking_confidence=0) as pose:
        frame = cv2.imread(image_path)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = results.pose_landmarks.landmark
            return landmarks
        except:
            return []
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def generate_df(folder_path:str = "", output_path:str = "") -> None:
    landmark_list = []
    count = 0
    for file in os.listdir(folder_path):
        landmarks = get_landmarks(os.path.join(folder_path, file))
        row = []
        if ("test" in output_path):
            row.append(file)
        for landmark in landmarks:
            row.append(landmark.x)
            row.append(landmark.y)
            row.append(landmark.z)
        if ("test" not in output_path): 
            row.append(file[0])
        landmark_list.append(row)
        count += 1
        print(count)
    print("done")
    df = pd.DataFrame(landmark_list)
    df.to_csv(output_path, index=False)
    return landmark_list

def organize_file(input_path:str = "train/", output_path:str = "renamed/") -> None:
    count0 = 0
    count1 = 0

    for subjects in os.listdir(input_path):
        for category in os.listdir(os.path.join(input_path, subjects)):
            for sub_category in os.listdir(os.path.join(input_path, subjects, category)):
                idx = 0
                for png in os.listdir(os.path.join(input_path, subjects, category, sub_category)):
                    current_path = os.path.join(input_path, subjects, category, sub_category, png)

                    if (category == "fall"): 
                        label = 1
                        count1 += 1
                        filename = str(label) + "_" + sub_category.split("_")[1] + "_" + str(count1)
                    else: 
                        label = 0
                        count0 += 1
                        filename = str(label) + "_" + sub_category.split("_")[1] + "_" + str(count0)

                    print(current_path)
                    shutil.copyfile(current_path, os.path.join(output_path, filename) + ".png")
    
    print("done")


if __name__ == "__main__":
    generate_df("renamed/", "train.csv")
    generate_df("test/", "test.csv")
    # path = "test/1ece69f362.jpg"
    # get_landmarks(path)