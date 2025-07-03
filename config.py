# Configuration file for Progressive GAN
#
# Make sure to set the paths and parameters according to your setup.
#
#
#


DATASET = "/media/volume/100K/bdd100k_images_100k/bdd100k/images/100k/train"

# Amount of epochs
EPOCHS = 5 # sets the amount of epochs ~ 11hrs/ea

#
Z_DIM = 512  # Latent vector size
BASE_CHANNELS = 512  # Base number of channels
IMAGE_CHANNELS = 3  # Number of image channels (e.g., 3 for RGB)
BATCH_SIZE = 16  # Default batch size
LEARNING_RATE = 0.0002  # Default learning rate

#IMG H and W
IMG_SIZE = 256  # Size of the input images (assumed square)
IMG_W = 1280  # Width of the input images
IMG_H = 720  # Height of the input images
MAX_IMAGES = 20000  # Maximum number of images to use from the dataset
# Add more configuration options as needed

# Hardware configuration
GPU_AMOUNT = 1  # Number of GPUs to use
NUM_WORKERS = 8
