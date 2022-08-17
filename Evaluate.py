import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, euclidean
from torchvision import transforms
from itertools import permutations
import matplotlib
matplotlib.use('TKAgg')


def get_embedding_from_image(img, model):
    img = img.astype('float32') / 255.
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = normalize(img)
    img = img.unsqueeze(0)
    img = img.cuda()
    return model(img)


model_path = './checkpoints/epoch_100_encoder_loss_18.4133358001709.pth'
model = torch.load(model_path)
model = model.cuda()

animals = glob.glob('../ViT/validation_data_cutre/dogs/*') + glob.glob('../ViT/validation_data_cutre/cats/*')
animals = list(permutations(animals, r=2))
random.shuffle(animals)

with torch.no_grad():
    for a, b in animals:
        image_a = cv2.resize(cv2.imread(a), (128, 128))[:, :, ::-1]
        image_b = cv2.resize(cv2.imread(b), (128, 128))[:, :, ::-1]

        embedding_a = get_embedding_from_image(image_a.copy(), model).cpu().data.numpy()
        embedding_b = get_embedding_from_image(image_b.copy(), model).cpu().data.numpy()

        joined_image = np.hstack([image_a, image_b])

        print(f"Cosine distance: {cosine(embedding_a, embedding_b)}")
        print(f"Euclidean distance: {euclidean(embedding_a, embedding_b)}")
        print(f"Cosine similarity: {cosine_similarity(embedding_a, embedding_b)}")

        plt.imshow(joined_image)
        plt.show()
