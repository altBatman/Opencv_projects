import cv2 ,time

first_frame=None
video=cv2.VideoCapture(0)

while(True):
    check, frame= video.read() #frame from video
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #gray frame
    gray=cv2.GaussianBlur(gray,(21,21),0)
    #setting up the base frame
    if first_frame is None:
        first_frame = gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    thres_frame=cv2.threshold(delta_frame,30,255, cv2.THRESH_BINARY)[1]
    thres_frame=cv2.dilate(thres_frame,None,iterations=2)

    cnts=cv2.findContours(thres_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]

    for contour in cnts:
        if cv2.contourArea(contour)<2000:
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),3)

    #cv2.imshow("Gray Frame",gray)
    #cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thres_frame)
    cv2.imshow("COLOUR Frame",frame)


    key=cv2.waitKey(1)
    if key==ord('q'):
        break

video.release()
cv2.destroyAllWindows
