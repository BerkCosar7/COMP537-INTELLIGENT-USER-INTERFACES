import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

from fer import FER
from fer import Video
import pandas as pd
import pyttsx3

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None  # VideoWriter object

cap = cv2.VideoCapture(0)
tracking_active = True

# Variable to store the last detected direction
last_direction = "Forward"

while cap.isOpened() and tracking_active:
    success, image = cap.read()
    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if x < -10:
                tracking_active = False  # Stop tracking

            # Determine the current head pose direction
            if y < -10:
                current_direction = "Looking Left"
            elif y > 10:
                current_direction = "Looking Right"
            elif x < -10:
                current_direction = "Looking Down"
                current_direction = "Forward"  # Reset for non-trigger action
            elif x > 10:
                current_direction = "Looking Up"
                current_direction = "Forward"  # Reset for non-trigger action
            else:
                current_direction = "Forward"

            # Perform actions only on the state transition
            if current_direction != last_direction:
                if current_direction == "Looking Left":
                    pyautogui.press('left')
                elif current_direction == "Looking Right":
                    pyautogui.press('right')

            # Update the last direction
            last_direction = current_direction

            text = current_direction

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
    cv2.imshow('Head Pose Estimation', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

    if out is None and tracking_active:
        frame_rate = 20
        out = cv2.VideoWriter('output.avi', fourcc, frame_rate, (img_w, img_h))

    # Write frame to video
    if out is not None:
        out.write(image)

cap.release()

if out is not None:
    out.release()

cv2.destroyAllWindows()

if out is not None:
    face_detector = FER(mtcnn=True)
    video_file = "output.avi"
    processed_video = Video(video_file=video_file)
    processing_data = processed_video.analyze(face_detector, display=True)

    df = pd.DataFrame(processing_data)
    print(df)

    emotion_columns = ['angry0', 'disgust0', 'fear0', 'happy0', 'sad0', 'surprise0', 'neutral0']
    average_emotions = df[emotion_columns].mean()

    # Print the average emotions
    print("Average emotions over the entire video:")
    print(average_emotions)

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    max_emotion = average_emotions.idxmax()

    vocal_feedback = {
        'angry0': 'You seem angry. Maybe you should calm down and try again',
        'disgust0': 'You seem disgusted. That is a problem. You should be more neutral.',
        'fear0': 'You seem fearful. Dont put stress on yourself, they are audience not your enemies.',
        'happy0': 'You seem happy. That is usually a good sign, but be careful, if this is an serious presentation, maybe you should be more neutral',
        'sad0': 'You seem sad. You should be more energetic, it will help audience to focus what you are doing',
        'surprise0': 'You seem surprised. It is sometimes alright but it shouldnt be dominant emotion of your presentation.',
        'neutral0': 'You seem neutral. That is the ideal case. Good one you are ready.'
    }
    if max_emotion in vocal_feedback:
        feedback_text = vocal_feedback[max_emotion]
        engine.say(feedback_text)
        engine.runAndWait()
    else:
        print("Unable to determine the dominant emotion.")


