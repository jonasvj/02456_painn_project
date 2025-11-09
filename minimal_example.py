"""
Basic example of how to train the PaiNN model to predict the QM9 property
"internal energy at 0K". This property (and the majority of the other QM9
properties) is computed as a sum of atomic contributions.
"""
import torch
import argparse
from tqdm import trange
import torch.nn.functional as F
from src.data import QM9DataModule
from src.utils import EarlyStopping
from pytorch_lightning import seed_everything
from src.models import PaiNN, AtomwisePostProcessing


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0)

    # Data
    parser.add_argument('--target', default=7, type=int) # 7 => Internal energy at 0K
    parser.add_argument('--data_dir', default='data/', type=str)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_inference', default=1000, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--splits', nargs=3, default=[110000, 10000, 10831], type=int) # [num_train, num_val, num_test]
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
    parser.add_argument('--early_stopping_min_epochs', default=500, type=int)

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
            mae += F.l1_loss(preds, batch.y, reduction='sum')
            N += len(batch.y)
        mae /= N

    return mae


def main():
    args = cli()
    seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dm = QM9DataModule(
        target=args.target,
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

    unit_conversion = dm.unit_conversion[args.target]
    y_mean, y_std, atom_refs = dm.get_target_stats(
        remove_atom_refs=True, divide_by_atoms=True
    )

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

    pbar = trange(args.num_epochs)
    for epoch in pbar:
        
        painn.train()
        loss_epoch = 0.
        for batch in train_loader:
            batch = batch.to(device)

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
            loss_step = F.mse_loss(preds, batch.y, reduction='sum')

            loss = loss_step / len(batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_epoch += loss_step.detach().item()

        loss_epoch /= len(dm.data_train)
        val_mae = compute_mae(painn, post_processing, val_loader, device)

        pbar.set_postfix_str(
            f'Train loss: {loss_epoch:.3e}, '
            f'Val. MAE: {unit_conversion(val_mae):.3f}'
        )

        stop = early_stopping.check(painn, val_mae, epoch)
        if stop:
            print(f'Early stopping after epoch {epoch}.')
            break

    painn = (
        early_stopping.best_model if early_stopping.best_model is not None 
        else painn
    )

    test_mae = compute_mae(painn, post_processing, test_loader, device)
    print(f'Test MAE: {unit_conversion(test_mae):.3f}')


if __name__ == '__main__':
    main()