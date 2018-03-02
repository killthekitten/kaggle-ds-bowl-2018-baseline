from bowl_config import BowlConfig

class InferenceConfig(BowlConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()