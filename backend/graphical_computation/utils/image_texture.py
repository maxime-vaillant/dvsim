from PIL import Image


class ImageTexture:
    def __init__(self, context, path):
        self.context = context
        try:
            img = Image.open(path).convert('RGB').transpose(Image.FLIP_TOP_BOTTOM)
            r, g, b = img.split()
            img = Image.merge('RGB', (b, g, r))
            self.texture = self.context.texture(img.size, 3, img.tobytes())
            self.texture.build_mipmaps()
        except FileNotFoundError:
            raise Exception(f"Texture file not found: {path}")

    def use(self):
        self.texture.use()