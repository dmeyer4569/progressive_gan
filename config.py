# Configuration file for Progressive GAN
#
# Make sure to set the paths and parameters according to your setup.
#
#
#


DATASET = "/media/volume/100K/bdd100k_images_100k/bdd100k/images/100k/train"

# Amount of epochs
EPOCHS = 50 # sets the amount of epochs ~ 30min/ea

CHECKPOINT_PT = "runs/2025-07-07_19-43-35/epochs/model_epoch_49.pt" # Keep none if you have no checkpoint or don't want to resume from a checkpoint

#
IMAGE_CHANNELS = 3  # Number of image channels (e.g., 3 for RGB)
BATCH_SIZE = 8  # Default batch size
LEARNING_RATE = 0.0002  # Default learning rate

#IMG H and W
IMG_SIZE = 512  # Size of the input images (assumed square)
IMG_W = 1280  # Width of the input images
IMG_H = 720  # Height of the input images
MAX_IMAGES = 2500  # Maximum number of images to use from the dataset
# Add more configuration options as needed

# Hardware configuration
GPU_AMOUNT = 1  # Number of GPUs to use
NUM_WORKERS = 16
