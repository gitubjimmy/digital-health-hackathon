def train():

    import os
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    import torch
    import torch.nn as nn
    import torch.nn.functional as f
    from torch.utils.data import SubsetRandomSampler

    import config
    from models import get_model, get_optimizer_from_config, get_lr_scheduler_from_config
    from data_prep_utils import get_processed_data, get_loader
    from train_utils import RegressionTrainer, \
        visualize_regression, visualize_learning, visualize_early_stopping_epochs, \
        k_fold_early_stopping
    from utils import file_output

    dataset = get_processed_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # print("Calculating early stopping epoch from fold repeating...\n")
    #
    # num_epochs = config.EPOCH_PER_K_FOLD
    # repeat = config.K_FOLD_REPEAT
    # num_folds = config.NUM_K_FOLD
    #
    # early_stopping_epochs = k_fold_early_stopping(num_folds, num_epochs, repeat)
    #
    # visualize_early_stopping_epochs(
    #     early_stopping_epochs,
    #     title="KFold Average Early Stopping Epochs",
    #     figsize=(15, 12),
    #     filename="output_train_kfold_iter.png",
    #     show=False
    # )
    # early_stopping_epoch = (sum(early_stopping_epochs) // len(early_stopping_epochs)) + 1
    # file_output(f"KFold {repeat} repeating - average early stopping epoch: {early_stopping_epoch}")

    early_stopping_epoch = 100

    print("\n\n\nActual training...\n")

    net = get_model(num_layers=None, channels=[2048, 512])
    optimizer = get_optimizer_from_config(net)
    scheduler = get_lr_scheduler_from_config(optimizer)
    loader = get_loader(dataset, train=True)
    fitter = RegressionTrainer(
        net, nn.MSELoss(), optimizer, scheduler, epoch=early_stopping_epoch,
        snapshot_dir=os.path.join(checkpoint_dir, "finalize"),
        train_iter=loader, val_iter=loader,
        verbose=True, progress=False, log_interval=1
    )
    fitter.to(device)
    _, train_result = fitter.fit(split_result=True)
    best_epoch = train_result.index(min(train_result)) + 1

    visualize_learning(
        train_result, train_result,
        title=f"Whole Data Learning Curve", figsize=(12, 12),
        filename=f"output_train_whole.png", show=False
    )

    print("\n\n\nSaving model...\n")

    checkpoint = os.path.join(checkpoint_dir, "finalize", f'best_checkpoint_epoch_{str(best_epoch).zfill(3)}.pt')
    model_state_dict = torch.load(checkpoint)['model']
    torch.save(model_state_dict, os.path.join(checkpoint_dir, "best_model_weight.pt"))

    print("\n\n\nLoading best weight...\n")

    net.load_state_dict(model_state_dict)
    net.eval().to(device)

    print("\n\n\nEvaluating...\n")

    with torch.no_grad():
        prediction = []
        label = []

        for inputs, labels in get_loader(dataset, train=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            prediction.append(outputs)
            label.append(labels)

        label = torch.cat(label, dim=0).view(-1, 1)
        prediction = torch.cat(prediction, dim=0).view(-1, 1)

        # transform to original
        sc_val = dataset.y_proc.scaler_
        label = sc_val.inverse_transform(label.cpu().numpy())  # numpy
        prediction = sc_val.inverse_transform(prediction.cpu().numpy())  # numpy
        outputs = torch.from_numpy(prediction).to(device).view(-1, 1)  # pytorch
        labels = torch.from_numpy(label).to(device).view(-1, 1)  # pytorch

        mse = f.mse_loss(outputs, labels).item()
        mae = f.l1_loss(outputs, labels).item()

    r2 = r2_score(label, prediction)

    file_output(f'MSE: {mse:.4f}\tMAE: {mae:.4f}\tTrain R2 score: {r2:.4f}')

    visualize_regression(
        label, prediction, mse_score=mse, mae_score=mae, r2_score=r2,
        plot_max=200, plot_min=0, vmax=40., vmin=0., alpha=0.5, figsize=(15, 12),
        xlabel='Value', ylabel='Prediction', title='Year of Survival Regression',
        filename=f"output_train_regression.png", show=False
    )


if __name__ == '__main__':
    train()
