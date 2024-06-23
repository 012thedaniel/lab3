import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN


# Функція для демонстрації зображення з обрамленням виявлених облич
def display_faces(img, faces):
    # Відображення зображення
    plt.imshow(img)
    # Отримання поточної вісі для малювання прямокутників
    ax = plt.gca()
    for face in faces:
        # Отримання координат обрамлення обличчя
        x, y, width, height = face['box']
        # Створення прямокутника
        rect = plt.Rectangle((x, y), width, height, fill=False, color='red')
        # Додавання прямокутника на графік
        ax.add_patch(rect)
    # Вимкнення відображення осей координат
    plt.axis('off')
    # Відображення графіку
    plt.show()


if __name__ == '__main__':
    # Зчитування зображення за допомогою OpenCV
    image = cv2.imread('image.png')
    # Конвертація зображення в формат RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Ініціалізація детектора облич MTCNN
    detector = MTCNN()
    # Виявлення облич у зображенні за допомогою MTCNN
    detected_faces = detector.detect_faces(image_rgb)

    # Відображення зображення з виявленими обличчями
    display_faces(image_rgb, detected_faces)
