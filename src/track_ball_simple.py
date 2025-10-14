import cv2 as cv
import numpy as np


video_path = '../../videos/test2/test2_1A.mp4'
video = cv.VideoCapture(video_path)

# Generate a white circle on a blue background and get sift descriptors
ball_img = np.zeros((512, 512, 3), np.uint8)
ball_img[:] = (255, 0, 0)

ball_img = cv.circle(ball_img, (256, 256), 128, (255, 255, 255), -1)

ball_gray = cv.cvtColor(ball_img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp_ball, des_ball = sift.detectAndCompute(ball_img, None)

bf = cv.BFMatcher()


# Process and stream each frame and draw hough circles    
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # get best SIFT match
    kp_frame, des_frame = sift.detectAndCompute(gray, None)
    matches = bf.knnMatch(des_ball, des_frame, k=2)

    out = cv.drawMatchesKnn(ball_img, kp_ball, gray, kp_frame, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
       
    cv.imshow('Frame', out)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows()