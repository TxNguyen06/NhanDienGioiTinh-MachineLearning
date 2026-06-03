import cv2
import joblib
import numpy as np
from ultralytics import YOLO 
#Mô hình
print("Đang khởi động hệ thống YOLOv8-Face")
gender_model = joblib.load('gender_model.pkl')

face_model = YOLO('face_yolov8n.pt')

#Kết nối vs Camera điện thoạt
CAMERA_SOURCE = 'http://192.168.1.99:4747/video'
cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print("\nKhông thể kết nối tới DroidCam")
    exit()

print("Kết nối Camera thành công")

#Xử lí camera
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    #Xác định khuôn mặt
    results = face_model(frame, stream=True, verbose=False)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Lấy tọa độ khung vuông khuôn mặt
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            h_pad = int((y2 - y1) * 0.15)
            w_pad = int((x2 - x1) * 0.15)

            #Xây dựng toạ độ mới
            y1_new = max(0, y1 - h_pad)
            y2_new = min(frame.shape[0], y2 + h_pad)
            x1_new = max(0, x1 - w_pad)
            x2_new = min(frame.shape[1], x2 + w_pad)
            
            #Cắt khuôn mặt 
            crop_face = frame[y1_new:y2_new, x1_new:x2_new]
            if crop_face.shape[0] > 0 and crop_face.shape[1] > 0:
                try:
                    resized_face = cv2.resize(crop_face, (100, 100))
                    flattened_face = resized_face.flatten().reshape(1, -1)
                    
                    #Dự đoán phần trăm 
                    probabilities = gender_model.predict_proba(flattened_face)[0]
                    prediction = np.argmax(probabilities)
                    confidence_score = probabilities[prediction] * 100
                    
                    if prediction == 0:
                        display_text = f"Nam: {confidence_score:.1f}%"
                        color = (255, 0, 0)
                    else:
                        display_text = f"Nu: {confidence_score:.1f}%"
                        color = (0, 0, 255)
                    
                    #Vẽ khung vuông và ghi chữ
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, display_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    pass

    cv2.imshow("He thong Nhan dien Gioi tinh - YOLO Face", frame)
    print("Nhấn nút'x' để đóng camera")

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()