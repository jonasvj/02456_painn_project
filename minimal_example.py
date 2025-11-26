"""
Basic example of how to train the PaiNN model to predict the MD17 energy.
"""
import torch
import argparse
from tqdm import trange
import torch.nn.functional as F
from src.data.md17 import MD17DataModule
from src.utils import EarlyStopping
from pytorch_lightning import seed_everything
from src.models import PaiNN, AtomwisePostProcessing


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)

    # Data
    parser.add_argument('--molecule_name', default='ethanol', type=str)
    parser.add_argument('--data_dir', default='data_md17/', type=str)
    parser.add_argument('--batch_size_train', default=10, type=int)
    parser.add_argument('--batch_size_inference', default=100, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--splits', nargs=3, default=[1000, 100, 100], type=int) # [num_train, num_val, num_test]
    parser.add_argument('--subset_size', default=None, type=int)

    # Model
    parser.add_argument('--num_message_passing_layers', default=3, type=int)
    parser.add_argument('--num_features', default=128, type=int)
    parser.add_argument('--num_outputs', default=1, type=int)
    parser.add_argument('--num_rbf_features', default=20, type=int)
    parser.add_argument('--num_unique_atoms', default=100, type=int)
    parser.add_argument('--cutoff_dist', default=5.0, type=float)

    # Training
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--early_stopping_patience', default=30, type=int)
    parser.add_argument('--early_stopping_min_epochs', default=1000, type=int)

    args = parser.parse_args()
    return args


def compute_mae(painn, post_processing, dataloader, device):
    """
    Computes the mean absoulte error between PaiNN predictions and targets.
    """
    N = 0
    mae = 0
    painn.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            # MD17 usually stores energy in batch.energy
            target = batch.energy if hasattr(batch, 'energy') else batch.y
            if target.ndim == 1:
                target = target.unsqueeze(-1)

            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch,
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            mae += F.l1_loss(preds, target, reduction='sum')
            N += len(target)
        mae /= N

    return mae


def main():
    args = cli()
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dm = MD17DataModule(
        molecule_name=args.molecule_name,
        data_dir=args.data_dir,
        batch_size_train=args.batch_size_train,
        batch_size_inference=args.batch_size_inference,
        num_workers=args.num_workers,
        splits=args.splits,
        seed=args.seed,
        subset_size=args.subset_size,
    )
    dm.prepare_data()
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # Calculate statistics from training data
    print("Computing statistics from training data...")
    energies = []
    num_atoms_list = []
    for batch in train_loader:
        target = batch.energy if hasattr(batch, 'energy') else batch.y
        energies.append(target)
        num_atoms_list.append(batch.ptr[1:] - batch.ptr[:-1])
    energies = torch.cat(energies, dim=0)
    num_atoms = torch.cat(num_atoms_list, dim=0).float()
    
    avg_num_atoms = num_atoms.mean()
    y_mean = energies.mean() / avg_num_atoms
    y_std = energies.std() / avg_num_atoms
    
    # MD17 has fixed composition, so we can set atom_refs to 0 or handle it.
    # For simplicity, we'll just use 0s for now as we normalize with mean/std.
    # We need to know max atomic number for embedding size, or just use a safe large number.
    # MD17 molecules usually have C, H, O, N, etc. (Z < 20).
    atom_refs = torch.zeros(100, 1) 

    print(f"Mean atom energy: {y_mean.item():.4f}, Std atom energy: {y_std.item():.4f}")

    painn = PaiNN(
        num_message_passing_layers=args.num_message_passing_layers,
        num_features=args.num_features,
        num_outputs=args.num_outputs, 
        num_rbf_features=args.num_rbf_features,
        num_unique_atoms=args.num_unique_atoms,
        cutoff_dist=args.cutoff_dist,
    )
    post_processing = AtomwisePostProcessing(
        args.num_outputs, y_mean, y_std, atom_refs
    )

    painn.to(device)
    post_processing.to(device)

    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_epochs=args.early_stopping_min_epochs,
    )

    optimizer = torch.optim.AdamW(
        painn.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs*len(train_loader)
    )

    # Keep history for quick visualization after training.
    train_losses = []
    val_maes = []

    pbar = trange(args.num_epochs)
    for epoch in pbar:
        
        painn.train()
        loss_epoch = 0.
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.energy if hasattr(batch, 'energy') else batch.y
            if target.ndim == 1:
                target = target.unsqueeze(-1)

            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            loss_step = F.mse_loss(preds, target, reduction='sum')
            
            # ELBO Loss = MSE + beta * KL
            # beta weighting: 1 / num_training_samples is a common choice
            beta = 1.0 / len(dm.data_train)
            kl_div = painn.kl_divergence
            loss = (loss_step + beta * kl_div) / len(target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_epoch += loss_step.detach().item()

        loss_epoch /= len(dm.data_train)
        val_mae = compute_mae(painn, post_processing, val_loader, device)

        # Track metrics for plots
        train_losses.append(loss_epoch)
        val_maes.append(val_mae.item())

        pbar.set_postfix_str(
            f'Train loss: {loss_epoch:.3e}, '
            f'Val. MAE: {val_mae:.3f}'
        )

        stop = early_stopping.check(painn, val_mae, epoch)
        if stop:
            print(f'Early stopping after epoch {epoch}.')
            break

    painn = (
        early_stopping.best_model if early_stopping.best_model is not None 
        else painn
    )
    print(f'Best epoch: {early_stopping.best_epoch}')
    print(f'Best val. MAE: {early_stopping.best_loss}')

    test_mae = compute_mae(painn, post_processing, test_loader, device)
    print(f'Test MAE: {test_mae:.3f}')

    # Save quick plots (loss/MAE curves and preds vs targets on test set).
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; skipping plots.')
        return

    # Loss / MAE curves.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(train_losses, label='Train loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Train loss')
    ax[0].grid(True)
    ax[1].plot(val_maes, label='Val MAE')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MAE')
    ax[1].set_title('Validation MAE')
    ax[1].grid(True)
    fig.tight_layout()
    fig.savefig('training_curves.png', dpi=150)
    plt.close(fig)

    # Predictions vs targets on test set.
    preds_list = []
    targets_list = []
    painn.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            target = batch.energy if hasattr(batch, 'energy') else batch.y
            if target.ndim == 1:
                target = target.unsqueeze(-1)
            
            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch,
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            preds_list.append(preds.cpu())
            targets_list.append(target.cpu())

    preds_all = torch.cat(preds_list, dim=0).squeeze(-1)
    targets_all = torch.cat(targets_list, dim=0).squeeze(-1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(targets_all, preds_all, s=5, alpha=0.5)
    min_val = torch.min(torch.cat([targets_all, preds_all])).item()
    max_val = torch.max(torch.cat([targets_all, preds_all])).item()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title('Pred vs Target (test)')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig('pred_vs_target_test.png', dpi=150)
    plt.close(fig)

    print('Saved plots: training_curves.png, pred_vs_target_test.png')


if __name__ == '__main__':
    main()
