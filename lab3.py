import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN


def display_faces(img, faces):
    plt.imshow(img)
    ax = plt.gca()
    for face in faces:
        x, y, width, height = face['box']
        rect = plt.Rectangle((x, y), width, height, fill=False, color='red')
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    image_path = 'image.png'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    result = detector.detect_faces(image_rgb)

    display_faces(image_rgb, result)