import pyautogui as pg
from mediapipe import solutions as sl
import cv2 as cv

cam = cv.VideoCapture(0)  # cam
face_mesh = sl.face_mesh.FaceMesh(refine_landmarks=True)
hands = sl.hands.Hands()
screen_w, screen_h = pg.size()
print(screen_w, screen_h)
while True:  # main loop
    _, frame = cam.read()   # read cam
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    face_landmark = output.multi_face_landmarks
    frame_h, frame_w, frame_l = frame.shape
    print(face_landmark)
    if face_landmark:
        landmark = face_landmark[0].landmark
        for id, landmark in enumerate(landmark[474:478]):
            x = int(landmark.x * (frame_w * 3))
            y = int(landmark.y * (frame_h * 2))
            cv.circle(frame, (x, y), 3, (0, 0, 255))
            if id == 1:
                pg.moveTo(x, y)
                print(f"face value", {x}, {y})
    cv.imshow('My Mouse', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv.destroyAllWindows()