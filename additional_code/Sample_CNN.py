import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def generate_image(size=(128, 128), gap=False, closed_channel=False):
    image = Image.fromarray(np.random.randint(0, 256, size, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    if gap:
        if np.random.rand() > 0.5:
            y = np.random.randint(size[1] // 4, 3 * size[1] // 4)
            draw.line((0, y, size[0], y), fill=0, width=3)
        else:
            x = np.random.randint(size[0] // 4, 3 * size[0] // 4)
            draw.line((x, 0, x, size[1]), fill=0, width=3)
    if closed_channel:
        start_point = (np.random.randint(20, size[0]-20), np.random.randint(20, size[1]-20))
        end_point = (start_point[0] + np.random.randint(10, 30), start_point[1] + np.random.randint(10, 30))
        draw.line([start_point, end_point], fill=255, width=np.random.randint(1, 4))
    return np.array(image)

def save_images(num_images, base_dir):
    for subdir in ['train', 'validation', 'test']:
        path = os.path.join(base_dir, subdir)
        for label in ['0', '1']:
            os.makedirs(os.path.join(path, label), exist_ok=True)
    for i in range(num_images):
        gap = np.random.rand() > 0.5
        closed_channel = np.random.rand() > 0.5
        img = generate_image(gap=gap, closed_channel=closed_channel)
        label = '1' if (gap or closed_channel) else '0'
        if i < num_images * 0.7:
            subdir = 'train'
        elif i < num_images * 0.85:
            subdir = 'validation'
        else:
            subdir = 'test'
        filename = f"{i}.png"
        path = os.path.join(base_dir, subdir, label, filename)
        Image.fromarray(img).save(path)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(base_dir):
    datagen = ImageDataGenerator(rescale=1./255)
    train_generator = datagen.flow_from_directory(
        os.path.join(base_dir, 'train'),
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=20,
        class_mode='binary'
    )
    validation_generator = datagen.flow_from_directory(
        os.path.join(base_dir, 'validation'),
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=20,
        class_mode='binary'
    )
    test_generator = datagen.flow_from_directory(
        os.path.join(base_dir, 'test'),
        target_size=(128, 128),
        color_mode='grayscale',
        batch_size=20,
        class_mode='binary',
        shuffle=False  # Important for consistent results
    )
    model = build_model()
    model.fit(train_generator, epochs=10, validation_data=validation_generator)
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    
    # Predict and visualize images
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images).flatten()
    predictions = np.round(predictions)
    

    # Select 5 random samples to display
    indices = np.random.choice(range(len(test_labels)), 5, replace=False)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.imshow(test_images[idx].reshape(128, 128), cmap='gray')
        ax.set_title(f"Actual: {int(test_labels[idx])}\nPredicted: {int(predictions[idx])}")
        ax.axis('off')
    plt.show()
    return test_acc

# Usage
base_dir = 'synthetic_dataset'
save_images(300, base_dir)
test_accuracy = train_and_evaluate_model(base_dir)
