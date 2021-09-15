def train():

    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    import torch.utils.data
    import torch
    import torchinfo

    import config
    from models.model import get_model
    from data_prep_utils import get_processed_data, get_loader
    from train_utils import RegressionTrainer
    from utils import file_output

    dataset = get_processed_data()

    net = get_model()
    summary = torchinfo.summary(net, dataset[0][0].shape, verbose=0)  # 1: print, 0: return string
    print("\nModel Summary:\n\n```\n{}\n```\n".format(summary))  # for raw markdown

    criterion = torch.nn.MSELoss()
    optimizer_class = torch.optim.Adam
    optimizer_options = dict(lr=1e-3)
    scheduler_class = torch.optim.lr_scheduler.StepLR
    scheduler_options = dict(step_size=10)
    fold_count = config.NUM_K_FOLD
    num_epochs = config.EPOCH_PER_K_FOLD
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)

    def initialize_trainer(fold, train_loader=None, val_loader=None):
        n = get_model()
        o = optimizer_class(n.parameters(), **optimizer_options)
        s = scheduler_class(o, **scheduler_options)
        t = RegressionTrainer(
            n, criterion, o, s, epoch=num_epochs,
            snapshot_dir=os.path.join(checkpoint_dir, f"fold_{fold}"),
            train_iter=train_loader, val_iter=val_loader,
            log_interval=1, progress=False
        )
        # t.to(torch.float32)
        t.to(device)
        return t

    kf = KFold(n_splits=fold_count, random_state=0, shuffle=True)
    result = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n<Fold {fold}>")

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = get_loader(dataset, sampler=train_sampler)
        val_loader = get_loader(dataset, sampler=val_sampler)

        with initialize_trainer(fold, train_loader, val_loader) as trainer:
            fold_result = trainer.run(train_loader, val_loader)
            fold_result = np.array(fold_result)
            train_result = list(fold_result[:, 0])
            test_result = list(fold_result[:, 1])
            best_loss = trainer.best_loss
            result[fold] = dict(train_result=train_result, test_result=test_result, best_loss=best_loss)
            # Plot Learning Curve to check overfitting
            plt.figure(figsize=(12, 12))
            plt.plot(range(1, num_epochs + 1), train_result, label='Training Loss')
            plt.plot(range(1, num_epochs + 1), test_result, label='Test Loss')
            min_val_loss = test_result.index(min(test_result)) + 1
            plt.axvline(min_val_loss, linestyle='--', color='r', label='Early Stopping Checkpoint')
            plt.title(f"Fold {fold} Learning Curve")
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"output_train_fold_{fold}.png")

    best_fold = 0
    best_loss = float('inf')
    for fold, fold_result in result.items():
        if fold_result['best_loss'] < best_loss:
            best_fold = fold
            best_loss = fold_result['best_loss']

    checkpoint = glob.glob(os.path.join(checkpoint_dir, f'fold_{best_fold}', 'best_checkpoint_epoch_*.pt'))[-1]

    net = get_model()
    net.load_state_dict(torch.load(checkpoint)['model'])
    net.to(device)

    with torch.no_grad():
        prediction = []
        Y_train_r2 = []
        running_loss = 0.0

        loader = get_loader(dataset, train=False)
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.to(torch.float32)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            prediction.append(outputs)
            Y_train_r2.append(labels)
            running_loss += loss.item() * labels.shape[0]

        prediction = torch.cat(prediction, dim=0).view(-1, 1).numpy()
        Y_train_r2 = torch.cat(Y_train_r2, dim=0).view(-1, 1).numpy()
        sc_val = dataset.y_proc.scaler_
        prediction_train = sc_val.inverse_transform(prediction)
        Y_train_r2 = sc_val.inverse_transform(Y_train_r2)
        r2_train = r2_score(Y_train_r2, prediction_train)
        loss_train = running_loss / len(dataset)

        file_output('Train MSE: {:.4f}\tTrain R2 score: {:.4f}'.format(loss_train, r2_train))

    plt.figure(figsize=(15,12))

    plt.plot([0, 200], [0, 200], color='black')
    plt.scatter(
        Y_train_r2, prediction_train,
        c=np.abs(Y_train_r2 - prediction_train), vmax=2.5, vmin=-0., cmap='jet', alpha=0.5
    )
    plt.colorbar(label='Error')

    plt.title('logP Regression')
    plt.xlabel('logP Value')
    plt.ylabel('Prediction')

    plt.text(
        100, 0,
        f'Whole MSE: {loss_train:.4f}\nWhole R2 score: {r2_train:.4f}',
        fontsize=25,
        bbox={'boxstyle': 'square', 'ec': (1,1,1), 'fc': (1,1,1), 'linestyle': '--', 'color': 'black'}
    )

    plt.grid(True)
    plt.minorticks_on()

    plt.savefig(f"output_train_whole.png")


if __name__ == '__main__':
    train()
