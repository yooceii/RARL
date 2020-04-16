import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


class Transform(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, color_jitter_magnitude, input_shape):
        super(Transform, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.decode = ops.ImageDecoder(output_type = types.RGB)
        self.RandomResizedCrop = ops.RandomResizedCrop(size=input_shape[-2:], random_area=(0.8,1.0))
        self.RandomFlip = ops.Flip(horizontal=1, vertical=1)
        self.RandomRotation = ops.Rotate(angle=90)
        # Check BrightnessContrast()
        self.RandomBrightness = ops.Brightness(brightness=0.8*color_jitter_magnitude)
        self.RandomContrast = ops.Contrast(contrast=0.8*color_jitter_magnitude)
        self.RandomHSV = ops.Hsv(saturation=0.8*color_jitter_magnitude, hue=0.8*color_jitter_magnitude)
        
    def define_graph(self):
        images = self.decode(self.images)
        images = self.RandomResizedCrop(images)
        images = self.RandomFlip(images)
        images = self.RandomRotation(images)
        if np.random.rand() <= 0.8:
            images = self.RandomBrightness(images)
            images = self.RandomContrast(images)
            images = self.RandomHSV(images)
        
        return images

    def add_image(self, images):
        self.images = images