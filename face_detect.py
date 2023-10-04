
import cv2 as cv

img_path = r'C:\\Users\Achref\Desktop\\Faces\\train\Benzima\\group 1.jpg'
haar_cascade_path = 'haar_face.xml'
img = cv.imread(img_path)
if img is None:
    print("Erreur lors du chargement de l'image.")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier(haar_cascade_path)

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

def display_detected_faces(image, faces):
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cv.putText(image, 'Visage', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if len(faces_rect) > 0:
    print(f'Nombre de visages détectés : {len(faces_rect)}')
    display_detected_faces(img, faces_rect)
else:
    print("Aucun visage détecté dans l'image.")

max_height = 800
if img.shape[0] > max_height:
    scale_factor = max_height / img.shape[0]
    img = cv.resize(img, (int(img.shape[1] * scale_factor), max_height))

cv.imshow('Visages détectés', img)

cv.waitKey(0)
cv.destroyAllWindows()
