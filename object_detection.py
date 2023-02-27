import cv2
import numpy as np
import pyttsx3
import count


def AI_speak(something):
    engine = pyttsx3.init()
    engine.say(something)
    engine.runAndWait()

cam = cv2.VideoCapture('http://192.168.177.184:8080/video')
result, image = cam.read()
if result:

    cv2.imshow("Image", image)
    cv2.imwrite("detect.png", image)
    net = cv2.dnn.readNet('yolov3-spp.weights', 'yolov3-spp.cfg')

    classes = []

    with open("coco.names.txt", "r") as f:
        classes = f.read().splitlines()
        img = cv2.imread('detect.png')

    # while True:
    # _, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

count_array=[]
if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)
        count_array.append(label)
    my_dict = {i: count_array.count(i) for i in count_array}
    print(my_dict)
    AI_speak(my_dict)
    cv2.imshow('Image', img)
    cam.release()
    cv2.destroyAllWindows()
else:
    AI_speak("Nothing detected")