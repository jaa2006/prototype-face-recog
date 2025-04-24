import cv2

# Load classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Nyalakan webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah ke grayscale (agar deteksi lebih ringan)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Gambar kotak & teks
    for (x, y, w, h) in faces:
        # Kotak di wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Tambahkan info
        cv2.putText(frame, "Nama: Zulfikar", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Role: DevOps", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Project: Face Recognition v2", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    # Tampilkan hasil frame
    cv2.imshow("Face Recognition v2", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
