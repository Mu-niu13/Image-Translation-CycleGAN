import os
import shutil
import torch
from torchvision.utils import save_image
from torch_fidelity import calculate_metrics


def save_images_to_dir(images, directory):
    """Save a batch of images to the specified directory."""
    os.makedirs(directory, exist_ok=True)
    for i, img in enumerate(images):
        img_path = os.path.join(directory, f"image_{i}.png")
        img = (img * 0.5 + 0.5).clamp(0, 1)
        save_image(img, img_path)


def calculate_fid_and_inception(real_images, generated_images):
    """Calculate FID and Inception Score."""
    real_dir = "./temp/real_monet_images"
    fake_dir = "./temp/generated_images"

    # save images to directories
    save_images_to_dir(real_images, real_dir)
    save_images_to_dir(generated_images, fake_dir)

    # calculate metrics IS and FID
    metrics = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        cuda=torch.cuda.is_available(),
        isc=True,  # Inception Score
        fid=True,  # FID
        verbose=False,
    )

    # clean up temporary directories
    shutil.rmtree(real_dir)
    shutil.rmtree(fake_dir)

    return metrics
