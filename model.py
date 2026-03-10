import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def list_audio_files(root: str) -> List[str]:
    files = []
    for path, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(path, name))
    files.sort()
    return files


class WaveformFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        sample_rate: int = 44100,
        duration: float = 5.0,
        normalize: bool = True,
    ):
        self.root = root
        self.files = list_audio_files(root)
        if len(self.files) == 0:
            raise RuntimeError(f"No audio files found in: {root}")

        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.files)

    def _load_audio(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # [C, T]

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav.squeeze(0)  # [T]

        # crop or pad to fixed length
        if wav.numel() > self.num_samples:
            start = torch.randint(0, wav.numel() - self.num_samples + 1, (1,)).item()
            wav = wav[start : start + self.num_samples]
        elif wav.numel() < self.num_samples:
            pad = self.num_samples - wav.numel()
            wav = F.pad(wav, (0, pad))

        if self.normalize:
            peak = wav.abs().max().clamp(min=1e-8)
            wav = wav / peak

        return wav

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        wav = self._load_audio(path)  # [T]
        return wav.unsqueeze(0)       # [1, T]


class ResidualStack1D(nn.Module):
    def __init__(self, channels: int, dilation_rates=(1, 3, 9)):
        super().__init__()
        self.blocks = nn.ModuleList()
        for d in dilation_rates:
            self.blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv1d(channels, channels, kernel_size=1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class RAVELikeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims=(64, 128, 256, 512),
        strides=(4, 4, 4, 2),
    ):
        super().__init__()
        assert len(hidden_dims) == len(strides)

        layers = []
        ch = in_channels
        for h, s in zip(hidden_dims, strides):
            layers.extend([
                nn.Conv1d(ch, h, kernel_size=2 * s, stride=s, padding=s // 2),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            ch = h

        self.body = nn.Sequential(*layers)
        self.to_mu = nn.Conv1d(ch, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(ch, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar


class RAVELikeDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims=(512, 256, 128, 64),
        strides=(2, 4, 4, 4),
    ):
        super().__init__()
        assert len(hidden_dims) == len(strides)

        self.in_proj = nn.Conv1d(latent_dim, hidden_dims[0], kernel_size=7, padding=3)

        blocks = []
        in_ch = hidden_dims[0]
        for i, s in enumerate(strides):
            out_ch = hidden_dims[i + 1] if i + 1 < len(hidden_dims) else hidden_dims[-1]
            blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose1d(
                        in_ch,
                        out_ch,
                        kernel_size=2 * s,
                        stride=s,
                        padding=s // 2,
                    ),
                    ResidualStack1D(out_ch),
                )
            )
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)

        self.waveform_head = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_ch, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, output_len: int = None) -> torch.Tensor:
        x = self.in_proj(z)
        x = self.blocks(x)
        x = self.waveform_head(x)

        if output_len is not None:
            if x.size(-1) > output_len:
                x = x[..., :output_len]
            elif x.size(-1) < output_len:
                x = F.pad(x, (0, output_len - x.size(-1)))

        return x


class RAVELikeVAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = RAVELikeEncoder(latent_dim=latent_dim)
        self.decoder = RAVELikeDecoder(latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, output_len: int = None) -> torch.Tensor:
        return self.decoder(z, output_len=output_len)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, output_len=x.size(-1))
        return x_hat, mu, logvar


def stft_mag(x: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
    window = torch.hann_window(win_length, device=x.device)
    spec = torch.stft(
        x.squeeze(1),  # [B, T]
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
    )
    return spec.abs()

#??? what is this
def multi_scale_stft_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    configs = [
        (2048, 512, 2048),
        (1024, 256, 1024),
        (512, 128, 512),
    ]
    loss = 0.0
    for n_fft, hop, win in configs:
        s_hat = stft_mag(x_hat, n_fft, hop, win)
        s = stft_mag(x, n_fft, hop, win)
        loss = loss + F.l1_loss(s_hat, s)
        loss = loss + F.l1_loss(torch.log(s_hat + 1e-7), torch.log(s + 1e-7))
    return loss / len(configs)


def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-4,
    wave_weight: float = 1.0,
    stft_weight: float = 1.0,
):
    recon_wave = F.l1_loss(x_hat, x)
    recon_stft = multi_scale_stft_loss(x_hat, x)
    recon = wave_weight * recon_wave + stft_weight * recon_stft

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    return loss, recon, kl


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


@torch.no_grad()
def save_reconstruction_examples(model, batch, device, out_dir, step, sample_rate):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    batch = batch.to(device)
    recon, _, _ = model(batch)

    batch = batch[:8].cpu()
    recon = recon[:8].cpu()

    torch.save(
        {
            "input": batch,
            "recon": recon,
            "sample_rate": sample_rate,
        },
        os.path.join(out_dir, f"recon_step_{step}.pt"),
    )

    for i in range(min(4, batch.size(0))):
        torchaudio.save(
            os.path.join(out_dir, f"step_{step}_input_{i}.wav"),
            batch[i],
            sample_rate,
        )
        torchaudio.save(
            os.path.join(out_dir, f"step_{step}_recon_{i}.wav"),
            recon[i].clamp(-1.0, 1.0),
            sample_rate,
        )


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Using device: {device}")

    dataset = WaveformFolderDataset(
        root=args.data_dir,
        sample_rate=args.sample_rate,
        duration=args.duration,
        normalize=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    model = RAVELikeVAE(latent_dim=args.latent_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and args.amp))

    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0

        for batch_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cuda" and args.amp)):
                recon, mu, logvar = model(batch)
                loss, recon_loss, kl = vae_loss(
                    batch,
                    recon,
                    mu,
                    logvar,
                    beta=args.beta,
                    wave_weight=args.wave_weight,
                    stft_weight=args.stft_weight,
                )

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kl += kl.item()
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = running_loss / batch_idx
                avg_recon = running_recon / batch_idx
                avg_kl = running_kl / batch_idx
                print(
                    f"Epoch [{epoch}/{args.epochs}] "
                    f"Step [{global_step}] "
                    f"Loss: {avg_loss:.6f} "
                    f"Recon: {avg_recon:.6f} "
                    f"KL: {avg_kl:.6f}"
                )

            if global_step % args.sample_every == 0:
                save_reconstruction_examples(
                    model,
                    batch,
                    device,
                    os.path.join(args.out_dir, "samples"),
                    global_step,
                    args.sample_rate,
                )

        ckpt_path = os.path.join(args.out_dir, f"rave_like_epoch_{epoch}.pt")
        save_checkpoint(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(args.out_dir, "rave_like_final.pt")
    save_checkpoint(
        {
            "epoch": args.epochs,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
        },
        final_path,
    )
    print(f"Training complete. Final model saved to: {final_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train a RAVE-like waveform VAE")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to audio dataset folder")
    parser.add_argument("--out_dir", type=str, default="./outputs_rave_like")

    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=5.0, help="Fixed audio clip length in seconds")

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-4, help="KL weight")
    parser.add_argument("--wave_weight", type=float, default=1.0)
    parser.add_argument("--stft_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--sample_every", type=int, default=200)

    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)