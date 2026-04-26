import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

# --------------------------------------------------
# 1. DEVICE
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------
CHECKPOINT_PATH = "best_resnet18_checkpoint.pth"

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
classes = checkpoint["class_names"]
num_classes = len(classes)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

print("Model loaded successfully")
print("Classes:", classes)

# --------------------------------------------------
# 3. DEFINE TRANSFORMS (MATCH TRAINING)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

# --------------------------------------------------
# 4. INITIALIZE WEBCAM & FACE DETECTOR
# --------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# 5. REAL-TIME INFERENCE LOOP
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        # OpenCV BGR → RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        try:
            face_tensor = transform(face_img).unsqueeze(0).to(device)
        except Exception as e:
            print("Transform error:", e)
            continue

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        emotion = classes[predicted.item()]
        conf_score = confidence.item() * 100

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{emotion} ({conf_score:.1f}%)"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------------------------------
# 6. CLEANUP
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()