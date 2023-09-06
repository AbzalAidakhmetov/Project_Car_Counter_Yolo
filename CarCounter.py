from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
#cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)

cap = cv2.VideoCapture('/home/theballer/Desktop/DS_Learning/Computer_Vision_Book/ObjectDetection/Object-Detection-101/Videos/cars.mp4')

model = YOLO('yolov8n.pt')


classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

mask = cv2.imread('/home/theballer/Desktop/DS_Learning/Computer_Vision_Book/ObjectDetection/r.png')

tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
limits = [400, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)
    
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #for bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            #for confidence
            conf = math.ceil((box.conf[0]*100))/100
            
            #for a class name
            c = int(box.cls[0])
            currentClass = classNames[c]
            if currentClass == 'car' and conf > 0.3:  
                #cv2.rectangle(img, (x1,y1),(x2, y2), (255,0,255), 3) 
                #cvzone.putTextRect(img, f'{classNames[c]} {conf}', (max(0, x1), max(35, y1)), scale = 0.6, thickness=1, offset=3)
                
                
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                
    resultTracker = tracker.update(detections)  
    
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)
    
    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        
        cv2.rectangle(img, (x1,y1),(x2, y2), (255,0,255), 3) 
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale = 0.9, thickness=1, offset=3)
        w, h = x2-x1, y2-y1
        cx, cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx,cy), 2, (255, 0, 255), cv2.FILLED)
        
        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[3]+20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)

    
    cvzone.putTextRect(img, f'Total Count: {len(totalCount)}', (50, 50))          
    cv2.imshow('Image', img)
    #cv2.imshow('ImageRegion', imgRegion)
    cv2.waitKey(1)