import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load mô hình đã huấn luyện
emotion_model = load_model('emotion_model.h5')

# Các nhãn cảm xúc theo thứ tự mà mô hình dự đoán
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Khởi tạo MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Khởi động webcam
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Đảo ngược hình ảnh để giống như gương
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Chuyển đổi sang RGB vì MediaPipe yêu cầu
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Lấy tọa độ khuôn mặt
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * frame.shape[1]), int(bboxC.ymin * frame.shape[0]), \
                             int(bboxC.width * frame.shape[1]), int(bboxC.height * frame.shape[0])

                # Cắt vùng khuôn mặt từ khung hình và chuyển đổi thành ảnh RGB 48x48
                face_roi = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face_roi, (48, 48)) / 255.0
                resized_face = np.expand_dims(resized_face, axis=0)  # Thêm chiều batch để phù hợp với đầu vào của mô hình

                # Dự đoán cảm xúc
                emotion_prediction = emotion_model.predict(resized_face)
                max_index = np.argmax(emotion_prediction)
                emotion_text = emotion_labels[max_index]

                # Vẽ bounding box và nhãn cảm xúc trên khung hình
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hiển thị khung hình
        cv2.imshow("Emotion Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

