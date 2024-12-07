import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from datasets import MonetDataset
from models import ProgressiveGenerator, ProgressiveDiscriminator
from losses import cycle_consistency_loss, identity_loss, gan_loss
from utils import calculate_fid_and_inception

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(f"Using device: {device}")


def train():
    resolutions = [16, 32, 64, 128, 256, 512]
    epochs_per_stage = 15
    lr = 0.0002
    beta1, beta2 = 0.5, 0.999

    lambda_cycle = 10
    lambda_identity = 5

    for resolution in resolutions:
        print(f"=== Training at resolution {resolution}x{resolution} ===")
        transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # adjust batch size for diff resolution
        if resolution <= 64:
            batch_size = 32
        elif resolution <= 256:
            batch_size = 16
        else:
            batch_size = 4

        # load CIFAR-10 and Monet datasets
        cifar_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar", train=True, download=True, transform=transform
        )
        monet_dataset = MonetDataset(root="./data/monet", transform=transform)

        cifar_loader = DataLoader(
            cifar_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        monet_loader = DataLoader(
            monet_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )

        # init models
        G_AtoB = ProgressiveGenerator(3, 3, resolution=resolution).to(device)
        G_BtoA = ProgressiveGenerator(3, 3, resolution=resolution).to(device)
        D_A = ProgressiveDiscriminator(3, resolution=resolution).to(device)
        D_B = ProgressiveDiscriminator(3, resolution=resolution).to(device)

        # optimizers
        optim_G = torch.optim.Adam(
            list(G_AtoB.parameters()) + list(G_BtoA.parameters()),
            lr=lr,
            betas=(beta1, beta2),
        )
        optim_D = torch.optim.Adam(
            list(D_A.parameters()) + list(D_B.parameters()), lr=lr, betas=(beta1, beta2)
        )

        for epoch in range(epochs_per_stage):
            G_AtoB.train()
            G_BtoA.train()
            D_A.train()
            D_B.train()

            epoch_loss_G = 0.0
            epoch_loss_D = 0.0

            for i, ((cifar_images, _), monet_images) in enumerate(
                zip(cifar_loader, monet_loader)
            ):
                cifar_images = cifar_images.to(device)
                monet_images = monet_images.to(device)

                # train generators
                optim_G.zero_grad()

                # monet->fake->monet cycle
                fake_monet = G_AtoB(cifar_images)  # cifar to monet
                cycle_cifar = G_BtoA(fake_monet)  # monet back to cifar

                # cifar->fake->cifar cycle
                fake_cifar = G_BtoA(monet_images)  # monet to cifar
                cycle_monet = G_AtoB(fake_cifar)  # cifar back to monet

                # identity passes
                identity_monet = G_AtoB(monet_images)  # monet through AtoB
                identity_cifar = G_BtoA(cifar_images)  # cifar through BtoA

                # generator loss
                loss_gan_AtoB = gan_loss(D_B(fake_monet), True)
                loss_gan_BtoA = gan_loss(D_A(fake_cifar), True)
                loss_cycle_A = cycle_consistency_loss(
                    cifar_images, cycle_cifar, lambda_cycle
                )
                loss_cycle_B = cycle_consistency_loss(
                    monet_images, cycle_monet, lambda_cycle
                )
                loss_identity_A = identity_loss(
                    monet_images, identity_monet, lambda_identity
                )
                loss_identity_B = identity_loss(
                    cifar_images, identity_cifar, lambda_identity
                )

                loss_G = (
                    loss_gan_AtoB
                    + loss_gan_BtoA
                    + loss_cycle_A
                    + loss_cycle_B
                    + loss_identity_A
                    + loss_identity_B
                )
                loss_G.backward()
                optim_G.step()

                # train discriminators
                optim_D.zero_grad()

                fake_cifar_detached = fake_cifar.detach()
                fake_monet_detached = fake_monet.detach()

                # D_A
                pred_real_A = D_A(cifar_images)
                pred_fake_A = D_A(fake_cifar_detached)
                loss_D_A = (
                    gan_loss(pred_real_A, True) + gan_loss(pred_fake_A, False)
                ) * 0.5
                loss_D_A.backward(retain_graph=True)

                # D_B
                pred_real_B = D_B(monet_images)
                pred_fake_B = D_B(fake_monet_detached)
                loss_D_B = (
                    gan_loss(pred_real_B, True) + gan_loss(pred_fake_B, False)
                ) * 0.5
                loss_D_B.backward()

                optim_D.step()

                loss_D = loss_D_A + loss_D_B

                epoch_loss_G += loss_G.item()
                epoch_loss_D += loss_D.item()

                # show 1 image pair
                if i == 0:

                    def denormalize(img):
                        return img * 0.5 + 0.5

                    original_image = cifar_images[0].detach().cpu()
                    generated_image = fake_monet[0].detach().cpu()

                    original_image = (
                        denormalize(original_image).permute(1, 2, 0).numpy()
                    )
                    generated_image = (
                        denormalize(generated_image).permute(1, 2, 0).numpy()
                    )

                    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                    axes[0].imshow(np.clip(original_image, 0, 1))
                    axes[0].set_title("Original CIFAR-10 Image")
                    axes[0].axis("off")

                    axes[1].imshow(np.clip(generated_image, 0, 1))
                    axes[1].set_title("Generated Monet-Style Image")
                    axes[1].axis("off")

                    plt.suptitle(
                        f"Epoch {epoch+1}, Resolution {resolution}x{resolution}"
                    )
                    plt.tight_layout()
                    plt.show()

                del (
                    fake_monet,
                    fake_cifar,
                    cycle_cifar,
                    cycle_monet,
                    identity_monet,
                    identity_cifar,
                )
                torch.cuda.empty_cache()

            print(
                f"Epoch {epoch+1}/{epochs_per_stage} | G Loss: {epoch_loss_G/len(cifar_loader):.4f} | "
                f"D Loss: {epoch_loss_D/len(cifar_loader):.4f}"
            )

        # get FID and IS
        G_AtoB.eval()
        required_samples = 320
        real_monet_images = []
        fake_monet_images = []

        monet_dataset_eval = MonetDataset(
            root="./data/monet",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        monet_loader_eval = DataLoader(
            monet_dataset_eval,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True,
        )
        monet_iter = iter(monet_loader_eval)
        cifar_iter = iter(cifar_loader)
        with torch.no_grad():
            while (
                len(real_monet_images) < required_samples
                and len(fake_monet_images) < required_samples
            ):
                real_batch = next(monet_iter)
                cifar_batch, _ = next(cifar_iter)

                real_monet_images.append(real_batch)
                fake_batch = G_AtoB(cifar_batch.to(device)).detach().cpu()
                real_monet_images = [torch.cat(real_monet_images, dim=0)]
                fake_monet_images.append(fake_batch)
                fake_monet_images = [torch.cat(fake_monet_images, dim=0)]

                if (
                    real_monet_images[0].size(0) >= required_samples
                    and fake_monet_images[0].size(0) >= required_samples
                ):
                    break

        real_monet_images = real_monet_images[0][:required_samples]
        fake_monet_images = fake_monet_images[0][:required_samples]

        metrics = calculate_fid_and_inception(real_monet_images, fake_monet_images)
        fid_score = metrics["frechet_inception_distance"]
        inception_score = metrics["inception_score_mean"]

        print(f"Metrics at Resolution {resolution}x{resolution}:")
        print(f"FID: {fid_score:.4f}")
        print(f"Inception Score: {inception_score:.4f}")


if __name__ == "__main__":
    train()
