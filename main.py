import cv2
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face = frame[y : y + h, x : x + w]

        try:
            emotion_analysis = DeepFace.analyze(
                face, actions=["emotion"], enforce_detection=False
            )
            dominant_emotion = emotion_analysis[0]["dominant_emotion"]

            cv2.putText(
                frame,
                dominant_emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )
        except Exception as e:
            print(f"Emotion detection failed: {e}")

    cv2.imshow("Face and Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
