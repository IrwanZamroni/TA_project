import cv2

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1920x1080 @ 30fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen
# Notice that we drop frames if we fall outside the processing time in the appsink element

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR  ! videobalance contrast=1.5 brightness=-.1  saturation=1.2 ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
window_title = "Face Detect"
video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
face_cascade = cv2.CascadeClassifier(
        "cascade.xml"
    )
while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hama = face_cascade.detectMultiScale(gray, 1.30, 2)

        for (x, y, w, h) in hama:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
           
        
        cv2.imshow(window_title, frame)
        
        keyCode = cv2.waitKey(10) & 0xFF
        # Stop the program on the ESC key or 'q'
        if keyCode == 27 or keyCode == ord('q'):
             break
                
video_capture.release()
cv2.destroyAllWindows()