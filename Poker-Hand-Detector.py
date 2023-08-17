from ultralytics import YOLO
import cv2
import math
import PokerHandFunction

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("playingCards.pt")
classNames = ['10C', '10D', '10H', '10S',
              '2C', '2D', '2H', '2S',
              '3C', '3D', '3H', '3S',
              '4C', '4D', '4H', '4S',
              '5C', '5D', '5H', '5S',
              '6C', '6D', '6H', '6S',
              '7C', '7D', '7H', '7S',
              '8C', '8D', '8H', '8S',
              '9C', '9D', '9H', '9S',
              'AC', 'AD', 'AH', 'AS',
              'JC', 'JD', 'JH', 'JS',
              'KC', 'KD', 'KH', 'KS',
              'QC', 'QD', 'QH', 'QS']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    hand = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            cv2.putText(img, f'{classNames[cls]}',(max(0, x1), max(35, y1)), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 3)
            if conf > 0.5:
                hand.append(classNames[cls])

    print(hand)
    hand = list(set(hand))
    print(hand)
    if len(hand) == 5:
        results = PokerHandFunction.findPokerHand(hand)
        print(results)

        cv2.putText(img, f'Your Hand: {results}',  (300, 75), cv2.FONT_HERSHEY_PLAIN, 3, (50, 200, 255), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
