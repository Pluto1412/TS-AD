import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.KAD_Disformer import KAD_Disformer
from utils.config import setup_seed
from utils.dataset import build_pretrain_dataset


def compute_freq_regularization(mask: torch.Tensor):
    # mask: [K, F]
    pairwise = torch.matmul(mask, mask.transpose(0, 1))
    overlap = pairwise.sum() - torch.diagonal(pairwise).sum()
    cover = ((mask.sum(dim=0) - 1.0) ** 2).mean()
    return overlap, cover


def main(args):
    setup_seed(args.seed)

    # Create checkpoint directory
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    model = KAD_Disformer(
        heads=args.heads,
        d_model=args.d_model,
        patch_size=args.patch_size,
        dropout=args.dropout,
        forward_expansion=args.forward_expansion,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
    )

    # --- Training ---
    train_dataset, train_stats = build_pretrain_dataset(
        args.data_path,
        win_len=args.win_len,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride,
        normalize_per_kpi=args.normalize_per_kpi,
    )
    total_series = sum(item["points"] for item in train_stats)
    total_samples = sum(item["samples"] for item in train_stats)
    print(
        f"Loaded {len(train_stats)} KPI series for pretraining "
        f"({total_series} points, {total_samples} training windows)."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=0.1)
    criterion = nn.MSELoss()

    print("Starting pre-training...")
    device = torch.device(args.device)
    model.to(device)
    model.train()
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, (seq_context, seq_history) in enumerate(progress_bar):
            optimizer.zero_grad()
            seq_context = seq_context.to(device)
            seq_history = seq_history.to(device)

            output, aux = model(seq_context, seq_history, return_aux=True)
            recon_loss = criterion(output, seq_context)
            overlap_loss, cover_loss = compute_freq_regularization(aux["mask"])
            loss = recon_loss + args.lambda_overlap * overlap_loss + args.lambda_cover * cover_loss
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                loss=loss.item(),
                recon=recon_loss.item(),
                overlap=overlap_loss.item(),
                cover=cover_loss.item(),
            )

        scheduler.step()

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Pre-training finished.")
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'model_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='KAD-Disformer Training')
    # Model parameters
    parser.add_argument('--heads', type=int, default=4, help='Number of heads in Multi-Head Attention')
    parser.add_argument('--d_model', type=int, default=64, help='Dimension of model')
    parser.add_argument('--patch_size', type=int, default=20, help='Patch size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--forward_expansion', type=int, default=1,
                        help='Forward expansion factor in FeedForward layer')
    parser.add_argument('--encoder_num_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--decoder_num_layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on (e.g., "cpu", "cuda")')

    # Data parameters
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/train.csv',
        help='Path to a single train.csv or a directory of KPI subdirectories containing train.csv files',
    )
    parser.add_argument('--win_len', type=int, default=20, help='Window length for dataset')
    parser.add_argument('--seq_len', type=int, default=100, help='Sequence length for dataset')
    parser.add_argument('--seq_stride', type=int, default=10, help='Sequence stride for dataset')
    parser.add_argument(
        '--normalize_per_kpi',
        action='store_true',
        help='Normalize each KPI series independently before windowing. Recommended for multi-KPI pretraining.',
    )

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader worker processes')
    parser.add_argument('--pin_memory', action='store_true', help='Enable pinned memory for faster host-to-device transfer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr_step_size', type=int, default=5, help='Step size for learning rate scheduler')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--lambda_overlap', type=float, default=1e-3, help='Weight for frequency overlap regularization')
    parser.add_argument('--lambda_cover', type=float, default=1e-3, help='Weight for frequency coverage regularization')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
