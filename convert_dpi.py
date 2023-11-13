from PIL import Image


class Image_dpi:
    def __init__(self, path):
        self.path = path
        self.img = Image.open(self.path)
        self.img = self.img.convert("RGB")
        self.img = self.img.resize((750, 750), Image.ANTIALIAS)
        self.img.save(self.path, dpi=(300, 300))
        self.img.close()
    
    def convert(self):
        self.img = Image.open(self.path)
        self.img = self.img.convert("RGB")
        self.img = self.img.resize((750, 750), Image.ANTIALIAS)
        self.img.save(self.path, dpi=(300, 300))
        self.img.close()
    
    def scale(self, size):
        self.img = Image.open(self.path)
        self.img = self.img.convert("RGB")
        self.img = self.img.resize((size, size), Image.ANTIALIAS)
        self.img.save(self.path, dpi=(500, 500))
        self.img.close()

    
