import torch
import torch.nn as nn
import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO

# Define the generator network
class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# Load the trained generator model (replace 'model.pth' with the actual path to the saved model)
latent_dim = 64
img_dim = 28 * 28
generator = Generator(latent_dim, img_dim)
generator.load_state_dict(torch.load('model.pth'))

# Function to generate and display images using the trained generator
def generate_image():
    z = torch.randn(1, latent_dim)
    gen_img = generator(z).view(28, 28).detach().numpy()
    gen_img = (gen_img + 1) / 2  # Rescale to [0, 1]
    image = Image.fromarray((gen_img * 255).astype('uint8'))
    image = image.resize((128, 128), Image.ANTIALIAS)
    image_tk = ImageTk.PhotoImage(image)
    label.config(image=image_tk)
    label.image = image_tk

# Create the Tkinter interface
root = tk.Tk()
root.title("GAN Image Generator")

label = tk.Label(root)
label.pack()

button = tk.Button(root, text="Generate Image", command=generate_image)
button.pack()

root.mainloop()

