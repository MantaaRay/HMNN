from math import log2
import torch

START_TRAIN_AT_IMG_SIZE = 32
DATASET = "MNIST"
NPZ_PATH = "data/data.npz"
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 16, 16, 16, 16, 16, 8, 4]
IMAGE_SIZE = 1024
CHANNELS_IMG = 1
Z_DIM = 512
IN_CHANNELS = 256
LAMBDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE / START_TRAIN_AT_IMG_SIZE)) + 1

PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 0
