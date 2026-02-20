"""
Standalone training script for GrapherNCA — run from terminal:

    # On Google Colab terminal (Tools > Terminal):
    python /content/drive/MyDrive/Experiments/Grapher_NCA/train_graphernca.py

    # Or locally with venv:
    source .venv/bin/activate
    python train_graphernca.py

Downloads ISIC 2018, then trains the chosen model with live terminal output.
"""

import os
import sys
import urllib.request
import zipfile
import argparse

# ─── Paths ────────────────────────────────────────────────────────────────────
# On Colab the script lives inside the Drive mount; adapt these if needed.
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR          # same folder as this script
DATASET_DIR  = os.path.join(PROJECT_ROOT, 'datasets', 'ISIC2018')
MODELS_DIR   = os.path.join(PROJECT_ROOT, 'Models')
REPO_DIR     = os.path.join(PROJECT_ROOT, 'M3D-NCA')

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# ─── Dataset download ─────────────────────────────────────────────────────────
ISIC_FILES = [
    (
        'ISIC2018_Task1-2_Training_Input',
        'https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip',
        'train_input.zip',
    ),
    (
        'ISIC2018_Task1_Training_GroundTruth',
        'https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip',
        'train_gt.zip',
    ),
    (
        'ISIC2018_Task1-2_Test_Input',
        'https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Test_Input.zip',
        'test_input.zip',
    ),
    (
        'ISIC2018_Task1_Test_GroundTruth',
        'https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Test_GroundTruth.zip',
        'test_gt.zip',
    ),
]


def _progress_bar(desc):
    """Returns a reporthook that prints a simple progress bar to stdout."""
    try:
        from tqdm import tqdm
        pbar = tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=desc)
        last = [0]

        def hook(count, block_size, total_size):
            if total_size > 0 and pbar.total is None:
                pbar.total = total_size
            downloaded = count * block_size
            pbar.update(downloaded - last[0])
            last[0] = downloaded

        hook.close = pbar.close
        return hook
    except ImportError:
        # Fallback: plain percentage printer
        last_pct = [-1]

        def hook(count, block_size, total_size):
            if total_size <= 0:
                print(f'\r{desc}: {count * block_size // 1024} KB downloaded', end='', flush=True)
                return
            pct = int(count * block_size * 100 / total_size)
            if pct != last_pct[0]:
                print(f'\r{desc}: {pct}%  ({count * block_size // 1024} / {total_size // 1024} KB)', end='', flush=True)
                last_pct[0] = pct
            if pct >= 100:
                print()

        hook.close = lambda: None
        return hook


def download_isic():
    """Download and extract ISIC 2018 dataset with visible progress."""
    print("\n=== ISIC 2018 Dataset Download ===")
    for folder, url, zipname in ISIC_FILES:
        dest_dir = os.path.join(DATASET_DIR, folder)
        if os.path.isdir(dest_dir):
            print(f'[skip] {folder} already exists.')
            continue

        zip_path = os.path.join(DATASET_DIR, zipname)
        print(f'\nDownloading {folder} ...')
        hook = _progress_bar(zipname)
        try:
            urllib.request.urlretrieve(url, zip_path, reporthook=hook)
        finally:
            hook.close()

        print(f'Extracting {zipname} ...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            try:
                from tqdm import tqdm
                for member in tqdm(members, desc='extracting', unit='file'):
                    zf.extract(member, DATASET_DIR)
            except ImportError:
                for i, member in enumerate(members):
                    zf.extract(member, DATASET_DIR)
                    if i % 500 == 0:
                        print(f'  extracted {i}/{len(members)} files...', flush=True)
        os.remove(zip_path)
        print(f'Done: {dest_dir}')

    print('\nDataset directory contents:')
    for entry in sorted(os.listdir(DATASET_DIR)):
        print(' ', entry)


# ─── Argument parsing ─────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Train GrapherNCA on ISIC 2018')
    parser.add_argument('--model', choices=['b1', 'm1', 'm2'], default='m1',
                        help='Model type: b1=BackboneNCA, m1=Pixel-Grapher, m2=Patch-Grapher')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', default='cuda:0',
                        help='PyTorch device string, e.g. cuda:0 or cpu or mps')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip dataset download (data already present)')
    parser.add_argument('--channel_n', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=9, help='KNN neighbours (m1/m2 only)')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size (m2 only)')
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # 1. Download data
    if not args.skip_download:
        download_isic()
    else:
        print('[skip_download] Skipping dataset download.')

    # 2. Imports (after sys.path is set up)
    print('\n=== Importing project modules ===')
    import torch
    from src.datasets.ISIC_Dataset import ISIC2018_Dataset
    from src.models.Model_BackboneNCA import BackboneNCA
    from src.models.Model_GrapherNCA_M1 import GrapherNCA_M1
    from src.models.Model_GrapherNCA_M2 import GrapherNCA_M2
    from src.losses.LossFunctions import DiceLoss
    from src.utils.Experiment import Experiment
    from src.agents.Agent_Med_NCA import Agent_Med_NCA

    # 3. Config
    model_type = args.model
    config = [{
        'img_path':   os.path.join(DATASET_DIR, 'ISIC2018_Task1-2_Training_Input'),
        'label_path': os.path.join(DATASET_DIR, 'ISIC2018_Task1_Training_GroundTruth'),
        'model_path': os.path.join(MODELS_DIR, f'GrapherNCA_single_{model_type}'),
        'device': args.device,
        'unlock_CPU': True,
        # Optimizer
        'lr': args.lr,
        'lr_gamma': 0.9999,
        'betas': (0.5, 0.5),
        # Training
        'save_interval': 10,
        'evaluate_interval': 10,
        'n_epoch': args.epochs,
        'batch_size': args.batch_size,
        # Model
        'channel_n': args.channel_n,
        'inference_steps': 10,
        'cell_fire_rate': 0.5,
        'input_channels': 3,
        'output_channels': 1,
        'hidden_size': 512,
        'train_model': 0,
        # Data
        'input_size': [(256, 256)],
        'data_split': [0.7, 0, 0.3],
    }]

    # 4. Device check
    device = torch.device(config[0]['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        print(f'WARNING: CUDA not available, falling back to CPU.')
        config[0]['device'] = 'cpu'
        device = torch.device('cpu')
    print(f'Device: {device}')

    # 5. Build model
    print(f'\n=== Building model: {model_type} ===')
    if model_type == 'b1':
        ca = BackboneNCA(
            config[0]['channel_n'], config[0]['cell_fire_rate'], device,
            hidden_size=config[0]['hidden_size'],
            input_channels=config[0]['input_channels'],
        ).to(device)
    elif model_type == 'm1':
        ca = GrapherNCA_M1(
            config[0]['channel_n'], config[0]['cell_fire_rate'], device,
            hidden_size=config[0]['hidden_size'],
            input_channels=config[0]['input_channels'],
            k=args.k,
        ).to(device)
    elif model_type == 'm2':
        ca = GrapherNCA_M2(
            config[0]['channel_n'], config[0]['cell_fire_rate'], device,
            hidden_size=config[0]['hidden_size'],
            input_channels=config[0]['input_channels'],
            k=args.k, patch_size=args.patch_size,
        ).to(device)

    total_params = sum(p.numel() for p in ca.parameters())
    print(f'Model: {model_type} | Parameters: {total_params:,}')

    ca = [ca]

    # 6. Dataset + Experiment
    print('\n=== Setting up dataset and experiment ===')
    dataset = ISIC2018_Dataset(input_channels=config[0]['input_channels'])
    agent   = Agent_Med_NCA(ca)
    exp     = Experiment(config, dataset, ca, agent)
    dataset.set_experiment(exp)
    exp.set_model_state('train')

    data_loader  = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=exp.get_from_config('batch_size')
    )
    loss_function = DiceLoss()

    print(f'Train samples: {len(dataset)}')
    print(f'Model saved to: {config[0]["model_path"]}')
    print(f'\n=== Starting training for {args.epochs} epochs ===\n')

    # 7. Train
    agent.train(data_loader, loss_function)


if __name__ == '__main__':
    main()
