from shred_s4_benchmark.utils.external import add_to_python_path
# from shred_s4_benchmark.models.mamba_model import MambaModel, MambawDecoder

try:
    # S4 imports
    add_to_python_path("shred_s4_benchmark/external/s4")
    from shred_s4_benchmark.models.s4_model import S4Model, S4wDecoder

    # SHRED imports
    add_to_python_path("shred_s4_benchmark/external/SHRED")
    from shred_s4_benchmark.external.shred.processdata import load_data, TimeSeriesDataset
    from shred_s4_benchmark.external.shred.models import SHRED
except ImportError as e:
    print("Please run setup_external.py with python -m shred_s4_benchmark.setup_external before running any other "
          "Python files")
    exit(1)


import argparse
from copy import deepcopy
import os
import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


# === Function for fitting S4 model ===
def fit(model, train_dataset, valid_dataset, batch_size=64, num_epochs=4000, lr=1e-3, optimizer=None, scheduler=None, verbose=False, patience=5):
    """

    THIS CODE IS MOSTLY COPIED FROM THE SHRED REPOSITORY WITH SLIGHT MODIFICATIONS TO ACCOUNT FOR S4 TRAINING MODIFICATIONS

    Function for training SHRED and S4 models
    """
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    criterion = torch.nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_error_list = []
    patience_counter = 0
    best_params = model.state_dict()
    for epoch in range(1, num_epochs + 1):

        for k, data in enumerate(train_loader):
            model.train()
            outputs = model(data[0])
            optimizer.zero_grad()
            loss = criterion(outputs, data[1])
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(valid_dataset.X)
                val_error = torch.linalg.norm(val_outputs - valid_dataset.Y)
                val_error = val_error / torch.linalg.norm(valid_dataset.Y)
                val_error_list.append(val_error)

            if verbose == True:
                print('Training epoch ' + str(epoch))
                print('Error (on scaled data) ' + str(val_error_list[-1]))
                if scheduler is not None:
                    print('Last learning rate ', scheduler.get_last_lr())

            if val_error == torch.min(torch.tensor(val_error_list)):
                patience_counter = 0
                best_params = deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter == patience:
                model.load_state_dict(best_params)
                return torch.tensor(val_error_list).cpu()

    model.load_state_dict(best_params)
    return torch.tensor(val_error_list).detach().cpu().numpy()


# === Data preprocessing ===
def preprocess_data(data, device, sequence_length, num_sensors):
    """

    THIS CODE IS MOSTLY COPIED FROM THE SHRED REPOSITORY WITH SLIGHT MODIFICATIONS

    Prepares datasets for training, validation, and testing.

    Selects random sensors, splits indices, scales data, creates sequences,
    converts to tensors on the specified device, and returns datasets and scaler.

    Args:
        data (np.ndarray): Time series data (time_steps × features).
        device (torch.device): Device for tensors.
        sequence_length (int): Length of input sequences.
        num_sensors (int): Number of sensors to select.

    Returns:
        train_dataset, valid_dataset, test_dataset, scaler
    """

    num_time_steps, num_spatial_locations = data.shape

    # Randomly select sensor locations
    sensor_indices = np.random.choice(num_spatial_locations, size=num_sensors, replace=False)

    # Generate training indices (sequence start positions)
    train_size = int((num_time_steps - sequence_length) * 0.75)
    train_indices = np.random.choice(num_time_steps - sequence_length, size=train_size, replace=False)

    # Create mask to separate validation and test indices from training
    mask = np.ones(num_time_steps - sequence_length, dtype=bool)
    mask[train_indices] = False
    available_indices = np.arange(num_time_steps - sequence_length)[mask]

    # Randomly select validation indices and test indices
    valid_size = int((num_time_steps - sequence_length) * 0.1)
    valid_indices = np.random.choice(available_indices, size=valid_size, replace=False)
    test_indices = np.setdiff1d(available_indices, valid_indices)

    # State information
    print(f"Training set size: {len(train_indices)}, validation set size: {len(valid_indices)}, test set size: {len(test_indices)}")

    # Fit scaler on training data and transform entire dataset
    scaler = MinMaxScaler()
    scaler.fit(data[train_indices])
    scaled_data = scaler.transform(data)

    # Prepare all input sequences for the model
    all_input_sequences = np.zeros((num_time_steps - sequence_length, sequence_length, num_sensors))
    for i in range(len(all_input_sequences)):
        all_input_sequences[i] = scaled_data[i:i + sequence_length, sensor_indices]

    # Convert input sequences and corresponding outputs to torch tensors
    def to_tensor(indices):
        inputs = torch.tensor(all_input_sequences[indices], dtype=torch.float32).to(device)
        outputs = torch.tensor(scaled_data[indices + sequence_length - 1], dtype=torch.float32).to(device)
        return inputs, outputs

    train_inputs, train_outputs = to_tensor(train_indices)
    valid_inputs, valid_outputs = to_tensor(valid_indices)
    test_inputs, test_outputs = to_tensor(test_indices)

    # Create dataset objects
    train_dataset = TimeSeriesDataset(train_inputs, train_outputs)
    valid_dataset = TimeSeriesDataset(valid_inputs, valid_outputs)
    test_dataset = TimeSeriesDataset(test_inputs, test_outputs)

    return train_dataset, valid_dataset, test_dataset, scaler


# === Setup optimizer for S4 model ===
def setup_optimizer(model, lr, weight_decay, epochs):
    """
    THIS IS COPIED FROM THE S4 REPOSITORY.
    NOT IMPORTED SINCE THE FILE EXAMPLE.PY WHERE THE FUNCTION EXISTS DOES NOT HAVE A MAIN GUARD.

    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler


# === Fit wrapper function ===
def train(model, model_name, train_ds, valid_ds, args):
    """Wrapper for the fit function"""

    def get_model_param(args, param):
        key = f"{model_name}_{param}"
        return getattr(args, key, None)

    start = time.time()
    print(f"Training {model_name} model")

    # Special optimizer and scheduler for s4
    if get_model_param(args, "optimize_lr"):
        optimizer, scheduler = setup_optimizer(
            model=model,
            lr=get_model_param(args, "lr"),
            weight_decay=get_model_param(args, "weight_decay"),
            epochs=args.epochs
        )
    else:
        optimizer, scheduler = None, None

    fit(
        model,
        train_ds,
        valid_ds,
        lr=1e-3 if get_model_param(args, "lr") is None else get_model_param(args, "lr"),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        optimizer=optimizer,
        scheduler=scheduler if get_model_param(args, "scheduler") else None,
        verbose=True,
        patience=args.patience
    )

    print(f"{model_name} time taken: ", time.time() - start)

    # Separator
    print("+" + "-" * 48 + "+")

    return model


def main(args):

    # Temporarily change working directory to enable file reads from SHRED repo
    current_dir = os.getcwd()
    os.chdir("shred_s4_benchmark/external/shred")

    # TODO: Perhaps filter out points which have a constant -1.8 value?
    data = load_data('SST')

    os.chdir(current_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds, valid_ds, test_ds, scaler = preprocess_data(data, device, args.sequence_length, args.num_sensors)
    num_spatial_locations = data.shape[1]

    # Initialize models
    """
    mamba = MambaModel(
        args.num_sensors,
        num_spatial_locations,
        d_conv=args.mamba_d_conv,
        d_model=args.mamba_d_model,
        d_state=args.mamba_d_state,
        expand=args.mamba_expand,
        n_layers=args.mamba_n_layers,
        dropout=args.mamba_dropout,
        prenorm=args.mamba_prenorm
    ).to(device)

    mambawdecoder = MambawDecoder(
        args.num_sensors,
        num_spatial_locations,
        d_conv=args.mambadecoder_d_conv,
        d_model=args.mambadecoder_d_model,
        d_state=args.mambadecoder_d_state,
        expand=args.mambadecoder_expand,
        l1=args.mambadecoder_l1,
        l2=args.mambadecoder_l2,
        n_layers=args.mambadecoder_n_layers,
        dropout=args.mambadecoder_dropout,
        prenorm=args.mambadecoder_prenorm
    )
    """
    shred = SHRED(
        args.num_sensors,
        num_spatial_locations,
        hidden_size=args.shred_hidden_size,
        hidden_layers=args.shred_hidden_layers,
        l1=args.shred_l1,
        l2=args.shred_l2,
        dropout=args.shred_dropout
    ).to(device)

    s4 = S4Model(
        d_input=args.num_sensors,
        d_output=num_spatial_locations,
        lr=args.s4_special_lr,
        d_model=args.s4_d_model,
        n_layers=args.s4_n_layers,
        dropout=args.s4_dropout,
        prenorm=args.s4_prenorm,
    ).to(device)

    s4wdecoder = S4wDecoder(
        d_input=args.num_sensors,
        d_output=num_spatial_locations,
        lr=args.s4decoder_special_lr,
        d_model=args.s4decoder_d_model,
        n_layers=args.s4decoder_n_layers,
        dropout=args.s4decoder_dropout,
        prenorm=args.s4decoder_prenorm,

        dropout_decoder=args.s4decoder_decoder_dropout,
        l1=args.s4decoder_l1,
        l2=args.s4decoder_l2
    ).to(device)

    """
    if args.mamba:
        mamba = train(mamba, "mamba", train_ds, valid_ds)
        
    if args.mambadecoder:
        mambawdecoder = train(mambawdecoder, "mambadecoder", train_ds, valid_ds)

    """

    if args.shred:
        shred = train(shred, "shred", train_ds, valid_ds, args)

    if args.s4:
        s4 = train(s4, "s4", train_ds, valid_ds, args)

    if args.s4decoder:
        s4wdecoder = train(s4wdecoder, "s4decoder", train_ds, valid_ds, args)

    # Testing
    print("\nTesting")
    test_ground_truth = scaler.inverse_transform(test_ds.Y.detach().cpu().numpy())

    """
    if args.mamba:
        test_recons = scaler.inverse_transform(mamba(test_ds.X).detach().cpu().numpy())
        print('(Mamba) Test Reconstruction Error (unscaled data): ')
        print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
        print()
        
    if args.mambadecoder:
        test_recons = scaler.inverse_transform(mambawdecoder(test_ds.X).detach().cpu().numpy())
        print('(Mamba + decoder) Test Reconstruction Error (unscaled data): ')
        print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
        print()
    """

    if args.shred:
        test_recons = scaler.inverse_transform(shred(test_ds.X).detach().cpu().numpy())
        print('(SHRED) Test Reconstruction Error (unscaled data): ')
        print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
        print()

    if args.s4:
        test_recons = scaler.inverse_transform(s4(test_ds.X).detach().cpu().numpy())
        print('(S4) Test Reconstruction Error (unscaled data): ')
        print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
        print()

    if args.s4decoder:
        test_recons = scaler.inverse_transform(s4wdecoder(test_ds.X).detach().cpu().numpy())
        print('(S4+decoder) Test Reconstruction Error (unscaled data): ')
        print(np.linalg.norm(test_recons - test_ground_truth) / np.linalg.norm(test_ground_truth))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mamba", action="store_true", help="Use the Mamba model")
    parser.add_argument("--mambadecoder", action="store_true", help="Use the Mamba decoder model")
    parser.add_argument("--shred", action="store_true", help="Use the SHRED model")
    parser.add_argument("--s4", action="store_true", help="Use the S4 model")
    parser.add_argument("--s4decoder", action="store_true", help="Use the S4+decoder model")

    parser.add_argument('--shred_dropout', type=float, default=0.1, help='SHRED dropout')
    parser.add_argument('--shred_hidden_layers', type=int, default=2, help='SHRED hidden layers')
    parser.add_argument('--shred_hidden_size', type=int, default=64, help='SHRED hidden state size')
    parser.add_argument('--shred_lr', type=float, default=1e-3, help='SHRED learning rate')
    parser.add_argument('--shred_l1', type=int, default=350, help='SHRED decoder layer 1 size')
    parser.add_argument('--shred_l2', type=int, default=400, help='SHRED decoder layer 2 size')

    parser.add_argument('--s4_dropout', type=float, default=0, help='S4 dropout')
    parser.add_argument('--s4_d_model', type=int, default=64, help='S4 model dimension')
    parser.add_argument('--s4_lr', type=float, default=1e-3, help='S4 learning rate for general parameters')
    parser.add_argument('--s4_n_layers', type=int, default=2, help='S4 layers')
    parser.add_argument('--s4_optimize_lr', action='store_true', help='turn on learning rate optimization for S4 model')
    parser.add_argument('--s4_prenorm', action='store_true', help='S4 prenorm')
    parser.add_argument('--s4_scheduler', action='store_true', help='Use a scheduler for the learning rate')
    parser.add_argument('--s4_special_lr', type=float, default=1e-3, help='S4 learning rate for special parameters, e.g., A matrix')
    parser.add_argument('--s4_weight_decay', type=float, default=0, help='S4 weight decay for general parameters')

    parser.add_argument('--s4decoder_decoder_dropout', type=float, default=0.1, help='S4 decoder dropout for decoder layers')
    parser.add_argument('--s4decoder_dropout', type=float, default=0, help='S4 decoder dropout for S4 layers')
    parser.add_argument('--s4decoder_d_model', type=int, default=64, help='S4 decoder model dimension')
    parser.add_argument('--s4decoder_lr', type=float, default=1e-3, help='S4 decoder learning rate for general parameters')
    parser.add_argument('--s4decoder_l1', type=int, default=350, help='S4 decoder layer 1 size')
    parser.add_argument('--s4decoder_l2', type=int, default=400, help='S4 decoder layer 2 size')
    parser.add_argument('--s4decoder_n_layers', type=int, default=2, help='S4 decoder layers')
    parser.add_argument('--s4decoder_optimize_lr', action='store_true', help='turn on learning rate optimization for S4 decoder model')
    parser.add_argument('--s4decoder_prenorm', action='store_true', help='S4 decoder prenorm')
    parser.add_argument('--s4decoder_scheduler', action='store_true', help='Use a scheduler for the learning rate')
    parser.add_argument('--s4decoder_special_lr', type=float, default=1e-3, help='S4 decoder learning rate for special parameters, e.g., A matrix')
    parser.add_argument('--s4decoder_weight_decay', type=float, default=0, help='S4 decoder weight decay for general parameters')

    parser.add_argument('--mamba_dropout', type=float, default=0)
    parser.add_argument('--mamba_d_conv', type=int, default=2)
    parser.add_argument('--mamba_d_model', type=int, default=64)
    parser.add_argument('--mamba_d_state', type=int, default=64)
    parser.add_argument('--mamba_expand', type=int, default=2)
    parser.add_argument('--mamba_lr', type=float, default=1e-3)
    parser.add_argument('--mamba_n_layers', type=int, default=2, help='Mamba layers')
    parser.add_argument('--mamba_prenorm', action='store_true', help='Mamba prenorm')

    parser.add_argument('--mambadecoder_decoder_dropout', type=float, default=0.1, help='Mamba decoder dropout for decoder layers')
    parser.add_argument('--mambadecoder_dropout', type=float, default=0, help='Mamba decoder dropout for Mamba layers')
    parser.add_argument('--mambadecoder_d_conv', type=int, default=2)
    parser.add_argument('--mambadecoder_d_model', type=int, default=64)
    parser.add_argument('--mambadecoder_d_state', type=int, default=64)
    parser.add_argument('--mambadecoder_expand', type=int, default=2)
    parser.add_argument('--mambadecoder_lr', type=float, default=1e-3)
    parser.add_argument('--mambadecoder_l1', type=int, default=350, help='Mamba decoder layer 1 size')
    parser.add_argument('--mambadecoder_l2', type=int, default=400, help='Mamba decoder layer 2 size')
    parser.add_argument('--mambadecoder_n_layers', type=int, default=2, help='S4 decoder layers')
    parser.add_argument('--mambadecoder_prenorm', action='store_true', help='Mamba decoder prenorm')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for SHRED and S4')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--num_sensors', type=int, default=3, help='number of sensors')
    parser.add_argument('--patience', type=int, default=5, help='Patience counter for training')
    parser.add_argument('--sequence_length', type=int, default=52, help='sequence length')

    args = parser.parse_args()

    main(args)
