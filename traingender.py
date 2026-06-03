import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle

#UTK Face
def load_utkface_dataset(data_path, max_samples=5000):
    images = []
    labels = []
    
    print("Đang xử lí Dataset UTKFace")
    if not os.path.exists(data_path):
        print(f"Không tìm thấy {data_path}!")
        return np.array(images), np.array(labels)

    count = 0
    for filename in os.listdir(data_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            if count >= max_samples: 
                break
            
            try:
                parts = filename.split('_')
                gender_label = int(parts[1])
                
                img_path = os.path.join(data_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                
                if img is not None:
                    img_resized = cv2.resize(img, (100, 100)) 
                    images.append(img_resized.flatten()) 
                    labels.append(gender_label)
                    count += 1
            except:
                continue
                
    print(f"Đã nạp thành công UTKFace.")
    return np.array(images), np.array(labels)

#AFAD
def load_afad_dataset(data_path, max_samples=5000):
    images = []
    labels = []
    
    print("Đang xử lí Dataset AFAD")
    if not os.path.exists(data_path):
        print(f"Không tìm thấy {data_path}! Bỏ qua AFAD.")
        return np.array(images), np.array(labels)

    count = 0
    for root, dirs, files in os.walk(data_path):
        if count >= max_samples:
            break
            
        if '111' in root:
            gender_label = 0
        elif '112' in root:
            gender_label = 1
        else:
            continue

        for filename in files:
            if count >= max_samples:
                break
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        img_resized = cv2.resize(img, (100, 100))
                        images.append(img_resized.flatten())
                        labels.append(gender_label)
                        count += 1
                except:
                    continue
                    
    print(f"Đã nạp thành công AFAD.")
    return np.array(images), np.array(labels)

#Gọi Dataset
PATH_UTKFACE = 'nhandienkhuonmat/UTKFace'
PATH_AFAD = 'AFAD-Lite/AFAD-Full' 

X_utk, y_utk = load_utkface_dataset(PATH_UTKFACE, max_samples=5000)
X_afad, y_afad = load_afad_dataset(PATH_AFAD, max_samples=5000)

#Trộn cả 2 mảng lại với nhau
X = np.concatenate((X_utk, X_afad), axis=0) if len(X_afad) > 0 else X_utk
y = np.concatenate((y_utk, y_afad), axis=0) if len(y_afad) > 0 else y_utk

#Tách riêng ảnh nam nữ
male_indices = np.where(y == 0)[0]
female_indices = np.where(y == 1)[0]

num_males = len(male_indices)
num_females = len(female_indices)

print(f"Thống kê trước khi cân bằng: Nam ({num_males} ảnh) - Nữ ({num_females} ảnh)")

#Tìm số để làm mốc cân bằng
min_samples = min(num_males, num_females)

#Lấy ngẫu nhiên số lượng ảnh
np.random.seed(42)
balanced_male_indices = np.random.choice(male_indices, min_samples, replace=False)
balanced_female_indices = np.random.choice(female_indices, min_samples, replace=False)

#Gộp lại thành mảng index cân bằng
balanced_indices = np.concatenate([balanced_male_indices, balanced_female_indices])

#Cập nhật lại X và y 
X = X[balanced_indices]
y = y[balanced_indices]

print(f"Đã cân bằng thành công: Nam ({min_samples} ảnh) == Nữ ({min_samples} ảnh). Tổng tập dữ liệu mới: {len(X)} ảnh.")

#Trộn dữ liệu
X, y = shuffle(X, y, random_state=42)

#Train 80% và Test 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

print(f"Tập huấn luyện (Train): {len(X_train)} mẫu")
print(f"Tập kiểm thử (Test): {len(X_test)} mẫu")

print("\nBắt đầu huấn luyện mô hình Logistic Regression...")

#Pipeline nén dữ liệu
model_pipeline = make_pipeline(
    StandardScaler(), 
    PCA(n_components=0.95, random_state=42), 
    LogisticRegression(max_iter=1000, random_state=42)
)

#Huấn luyện
model_pipeline.fit(X_train, y_train)

#Đánh giá
y_pred = model_pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

print(f"\nHuấn luyện mô hình xong. Mô hình có độ chính xác: {acc:.2f}%")
print("\nChi tiết báo cáo mô hình:")
print(classification_report(y_test, y_pred, target_names=['Nam (0)', 'Nu (1)']))

#Lưu file pkl
joblib.dump(model_pipeline, 'gender_model.pkl')
print("\nĐã lưu 'gender_model.pkl' thành công!")