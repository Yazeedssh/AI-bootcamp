import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
camera = cv2.VideoCapture(0)

while(True):
    read_ok ,frame = camera.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(-0,255,0))
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
    

camera.release()
cv2.destroyAllWindows()

