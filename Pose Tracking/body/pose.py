import cv2
import mediapipe as mp

mpPose=mp.solutions.pose
Poses=mpPose.Pose(False,False,True,0.8,0.8)
mpDraw=mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
  #  imgRGB = cv2.cvtColor(cv2.flip(frame,1) ,cv2.COLOR_BGR2RGB)
    imgRGB = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)
    results = Poses.process(imgRGB)
    #print([i for i in dir(results) if i.startswith('_') is False])
    if results.pose_landmarks:
        #for poseLM in results.pose_landmarks: 
        mpDraw.draw_landmarks(frame, results.pose_landmarks,
                mpPose.POSE_CONNECTIONS)
    cv2.imshow('image', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
