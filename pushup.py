import cv2
import numpy as np
import cvzone
from cvzone.PoseModule import PoseDetector
import math

class AngleFinder:
    def __init__(self, lmlist, p1, p2, p3, p4, p5, p6, drawPoints):
        self.lmlist = lmlist
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.drawPoints = drawPoints

    def calculate_angles(self):
        if len(self.lmlist) != 0:
            try:
                point1 = self.lmlist[self.p1]
                point2 = self.lmlist[self.p2]
                point3 = self.lmlist[self.p3]
                point4 = self.lmlist[self.p4]
                point5 = self.lmlist[self.p5]
                point6 = self.lmlist[self.p6]

                if len(point1) >= 2 and len(point2) >= 2 and len(point3) >= 2 and len(point4) >= 2 and len(point5) >= 2 and len(point6) >= 2:
                    x1, y1 = point1[:2]
                    x2, y2 = point2[:2]
                    x3, y3 = point3[:2]
                    x4, y4 = point4[:2]
                    x5, y5 = point5[:2]
                    x6, y6 = point6[:2]

                    leftHandAngle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
                    rightHandAngle = math.degrees(math.atan2(y6 - y5, x6 - x5) - math.atan2(y4 - y5, x4 - x5))

                    leftHandAngle = int(np.interp(leftHandAngle, [-170, 180], [100, 0]))
                    rightHandAngle = int(np.interp(rightHandAngle, [-50, 20], [100, 0]))

                    if self.drawPoints:
                        cv2.circle(img, (x1, y1), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x2, y2), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x2, y2), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x3, y3), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x3, y3), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x4, y4), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x4, y4), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x5, y5), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x5, y5), 15, (0, 255, 0), 6)
                        cv2.circle(img, (x6, y6), 10, (0, 255, 255), 5)
                        cv2.circle(img, (x6, y6), 15, (0, 255, 0), 6)

                        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 4)
                        cv2.line(img, (x4, y4), (x5, y5), (0, 0, 255), 4)
                        cv2.line(img, (x5, y5), (x6, y6), (0, 0, 255), 4)
                        cv2.line(img, (x1, y1), (x4, y4), (0, 0, 255), 4)

                    return leftHandAngle, rightHandAngle
            except IndexError:
                print("IndexError: Ensure all keypoints are detected.")
        return None, None

counter_pushups = 0
counter_dumbbell_curls = 0
direction_pushups = 0
direction_dumbbell_curls = 0
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('vid1.mp4')
pd = PoseDetector(trackCon=0.70, detectionCon=0.70)

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture image")
        continue

    img = cv2.resize(img, (1000, 500))
    cvzone.putTextRect(img, 'AI Exercise Counter', [345, 30], thickness=2, border=2, scale=2.5)
    pd.findPose(img, draw=0)
    lmlist, bbox = pd.findPosition(img, draw=0, bboxWithHands=0)

    angle_finder = AngleFinder(lmlist, 11, 13, 15, 12, 14, 16, drawPoints=True)
    left, right = angle_finder.calculate_angles()

    if left is not None and right is not None:
        # Check for dumbbell curls (example thresholds)
        if 0 <= left <= 90 and 0 <= right <= 90:
            if direction_dumbbell_curls == 0:
                counter_dumbbell_curls += 0.5
                direction_dumbbell_curls = 1
        else:
            direction_dumbbell_curls = 0

        # Check for push-ups (example thresholds)
        if left <= 70 and right <= 70:
            if direction_pushups == 0:
                counter_pushups += 0.5
                direction_pushups = 1
        else:
            direction_pushups = 0

            
        cv2.rectangle(img,(10, 10), (300, 120), (355, 0, 0), -1)
        cv2.putText(img, f"Push: {int(counter_pushups)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 7)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()