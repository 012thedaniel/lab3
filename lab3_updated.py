import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
import torch
import ssl


def extract_face(img, box, margin=20):
    x, y, width, height = box
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(img.shape[1], x + width + margin), min(img.shape[0], y + height + margin)
    return img[y1:y2, x1:x2]


def recognize_faces(img, faces, model):
    embeds = []
    for face in faces:
        face_img = extract_face(img, face['box'])
        face_img = cv2.resize(face_img, (160, 160))
        face_img = (face_img / 255.0).astype(np.float32)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)
        face_tensor = torch.tensor(face_img)
        embed = model(face_tensor)
        embeds.append(embed.detach().numpy())
    return embeds


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
    ssl._create_default_https_context = ssl._create_unverified_context
    image_path = 'image.png'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    result = detector.detect_faces(image_rgb)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    embeddings = recognize_faces(image_rgb, result, model)
    for i, embedding in enumerate(embeddings):
        print(f"Embedding for face {i + 1}: {embedding}")
    display_faces(image_rgb, result)
