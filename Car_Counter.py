import cv2
import numpy as np

cap = cv2.VideoCapture("Video.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

kernel = np.ones((5, 5), np.uint8)


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Sensor:
    def __init__(self, Coordinate1, Coordinate2, Sq_width, Sq_height):
        self.Coordinate1 = Coordinate1
        self.Coordinate2 = Coordinate2
        self.Sq_width = Sq_width
        self.Sq_height = Sq_height
        self.Mask_Area = abs(self.Coordinate2.x-Coordinate1.x) * abs(self.Coordinate2.y-self.Coordinate1.y)
        self.Mask = np.zeros((Sq_height, Sq_width, 1), np.uint8)
        cv2.rectangle(self.Mask, (self.Coordinate1.x, self.Coordinate1.y),(self.Coordinate2.x, self.Coordinate2.y), (255), cv2.FILLED)
        self.Statu = False
        self.Car_Number = 0


Sensor1 = Sensor(Coordinate(310, 180), Coordinate(420, 240), 1080, 250)


font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
while (1):
    ret, Square = cap.read()
    Sq_Cut = Square[350:600, 100:1180]

    Sq_WithoutBG = fgbg.apply(Sq_Cut)
    Sq_WithoutBG = cv2.morphologyEx(
        Sq_WithoutBG, cv2.MORPH_OPEN, kernel)

    ret, Sq_WithoutBG = cv2.threshold(
        Sq_WithoutBG, 127, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(
        Sq_WithoutBG, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    Result = Sq_Cut.copy()

    Img = np.zeros(
        (Sq_Cut.shape[0], Sq_Cut.shape[1], 1), np.uint8)

    Cnts_Count = 0
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if(w > 30 and h > 30):
            cv2.rectangle(Result, (x, y), (x+w, y+h), (0, 255, 0), 4)
            cv2.rectangle(Img, (x, y),(x+w, y+h), (255), cv2.FILLED)
            Cnts_Count += 1

    cv2.rectangle(Result, (Sensor1.Coordinate1.x, Sensor1.Coordinate1.y),
                  (Sensor1.Coordinate2.x, Sensor1.Coordinate2.y), (0, 0, 255), cv2.FILLED)

    Sensor1_Mask_Result = cv2.bitwise_and(Img, Img, mask=Sensor1.Mask)
    Sensor1_WhitePX_Count = np.sum(Sensor1_Mask_Result == 255)
    Sensor1_Rate = Sensor1_WhitePX_Count/Sensor1.Mask_Area

    if(Sensor1_Rate >= 0.75 and Sensor1.Result == False):
        cv2.rectangle(Result, (Sensor1.Coordinate1.x, Sensor1.Coordinate1.y),(Sensor1.Coordinate2.x, Sensor1.Coordinate2.y), (0, 255, 0), cv2.FILLED)
        Sensor1.Statu = True
    elif(Sensor1_Rate < 0.75 and Sensor1.Statu == True):
        cv2.rectangle(Result, (Sensor1.Coordinate1.x, Sensor1.Coordinate1.y),(Sensor1.Coordinate2.x, Sensor1.Coordinate2.y), (0, 0, 255), cv2.FILLED)
        Sensor1.Statu = False
        Sensor1.Car_Number += 1
    else:
        cv2.rectangle(Result, (Sensor1.Coordinate1.x, Sensor1.Coordinate1.y),(Sensor1.Coordinate2.x, Sensor1.Coordinate2.y), (0, 0, 255), cv2.FILLED)

    cv2.putText(Result, str(Sensor1.Car_Number),(Sensor1.Coordinate1.x+30, Sensor1.Coordinate1.y+50), font, 2, (0, 255, 255), 4)

    #cv2.imshow("Square", Square)
    #cv2.imshow("Sq_Cut",Sq_Cut)
    cv2.imshow("Sq_WithoutBG", Sq_WithoutBG)
    cv2.imshow("Image", Img)
    #cv2.imshow("Mask", Sensor1.Maske)
    cv2.imshow("Result", Result)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
