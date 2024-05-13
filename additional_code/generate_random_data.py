import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

def generate_image(size=(128, 128), gap=False, closed_channel=False):
    # Create a blank image and add random noise
    image = Image.fromarray(np.random.randint(0, 256, size, dtype=np.uint8))
    draw = ImageDraw.Draw(image)

    if gap:
        if np.random.rand() > 0.5:
            # Draw a horizontal gap
            y = np.random.randint(size[1] // 4, 3 * size[1] // 4)
            draw.line((0, y, size[0], y), fill=0, width=3)
        else:
            # Draw a vertical gap
            x = np.random.randint(size[0] // 4, 3 * size[0] // 4)
            draw.line((x, 0, x, size[1]), fill=0, width=3)

    if closed_channel:
        # Draw a closed-end channel
        start_point = (np.random.randint(20, size[0]-20), np.random.randint(20, size[1]-20))
        end_point = (start_point[0] + np.random.randint(10, 30), start_point[1] + np.random.randint(10, 30))
        draw.line([start_point, end_point], fill=255, width=np.random.randint(1, 4))

    return np.array(image)

def save_images(num_images, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    labels = []
    for i in range(num_images):
        gap = np.random.rand() > 0.5
        closed_channel = np.random.rand() > 0.5
        img = generate_image(gap=gap, closed_channel=closed_channel)
        label = f"{int(gap)}_{int(closed_channel)}"
        labels.append(label)
        filename = f"{i}_{label}.png"
        path = os.path.join(directory, filename)
        Image.fromarray(img).save(path)
    return labels

def visualize_images(images, labels):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 3))
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    plt.show()

# Example usage
labels = save_images(10, 'synthetic_dataset')
sample_images = [np.array(Image.open(f'synthetic_dataset/{i}_1_1.png')) for i in range(4)]
visualize_images(sample_images, labels[:4])
