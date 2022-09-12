class StableDiffusionSafetyCheckerDisable:
    def __init__(self, images=[], clip_input=[]):
        # return images, [False for _ in images]
        self.images = images
        pass

    def __iter__(self):
        yield self.images
        yield [False for _ in self.images]
