import cv2
import torch
from torchvision import transforms
from CNN import CNN_delay

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

classifier = CNN_delay()
classifier.load_state_dict(torch.load('weight_10 ideal edition.pth'))

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# cap = cv2.VideoCapture("video2.mp4")

#  Фронтальная камера
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(gray)

    if len(faces):
        x, y, w, h = max(faces, key=lambda x: x[-1] * x[-2])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = color[y:y + h, x:x + w]
        roi_tensor = transform(roi_gray)

        prediction = classifier(roi_tensor)
        label = emotion_labels[int(prediction[0].argmax())]
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('My emotion', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
