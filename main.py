from scipy.ndimage.filters import gaussian_filter
import pygame
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import time


label_names = requests.get("https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt").text.split("\n")

pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([784, 784])
# Run until the user asks to quit
running = True
screen.fill((0, 0, 0))
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model.h5", compile=False)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

data = np.ndarray(shape=(28, 28, 1), dtype=np.float32)


def predict_image(model, x):
  x = x.astype('float32')
  x = x / 255.0


  x = np.expand_dims(x, axis=0)


  image_predict = model.predict(x, verbose=0)
  print("Predicted Label: ", label_names[np.argmax(image_predict)])

  return image_predict


def plot_image(img):
  plt.imshow(np.squeeze(img))
  plt.xticks([])
  plt.yticks([])
  plt.show()



draw_last = (0, 0)
pressed_last = False
last_time = 0
while running:
    x, y = pygame.mouse.get_pos()

    # Did the user click the    window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:

            pygame.image.save(screen, "screenshot.jpg")


            # Replace this with the path to your image
            imgd = tf.keras.preprocessing.image.load_img('screenshot.jpg', target_size=(28, 28, 1),
                                                         color_mode="grayscale")
            img = tf.keras.preprocessing.image.img_to_array(imgd)
            img = gaussian_filter(img, sigma=0.5)

            predict_image(model, img)

    # if time.time() >= last_time + 1:
    #     pygame.image.save(screen, "screenshot.jpg")
    #
    #     # Replace this with the path to your image
    #     imgd = tf.keras.preprocessing.image.load_img('screenshot.jpg', target_size=(28, 28, 1),
    #                                                  color_mode="grayscale")
    #     img = tf.keras.preprocessing.image.img_to_array(imgd)
    #     img = gaussian_filter(img, sigma=0.5)
    #
    #     predict_image(model, img)
    #
    #     last_time = time.time()

    mouse_presses = pygame.mouse.get_pressed()
    if mouse_presses[0]:
        pygame.draw.circle(screen, (255, 255, 255), (x, y), 30)



    if mouse_presses[2]:
        screen.fill((0, 0, 0))
        drawn = []

    pressed_last = mouse_presses[0]
    draw_last = (x, y)

    # Flip the display
    pygame.display.flip()




# Done! Time to quit.
pygame.quit()


