def train():

    import os
    import glob
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    import torch
    import torch.nn as nn
    import torch.nn.functional as f
    from torch.utils.data import SubsetRandomSampler
    import torchinfo

    import config
    from models import get_model, get_optimizer_from_config, get_lr_scheduler_from_config
    from data_prep_utils import get_processed_data, get_loader
    from train_utils import RegressionTrainer, visualize_regression, visualize_learning
    from utils import file_output

    dataset = get_processed_data()

    net = get_model()
    summary = torchinfo.summary(net, dataset[0][0].shape, verbose=0)  # 1: print, 0: return string
    print("\n## Model Summary:  \n\n```\n{}\n```\n".format(summary))  # for raw markdown

    criterion = nn.MSELoss()
    num_folds = config.NUM_K_FOLD
    num_epochs = config.EPOCH_PER_K_FOLD
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_verbose = False

    def initialize_trainer(fold_count):
        n = get_model()
        o = get_optimizer_from_config(n)
        s = get_lr_scheduler_from_config(o)
        t = RegressionTrainer(
            n, criterion, o, s, epoch=num_epochs,
            snapshot_dir=os.path.join(checkpoint_dir, f"fold_{fold_count}"),
            # train_iter=train_loader, val_iter=val_loader,
            verbose=train_verbose, progress=False, log_interval=1
        )
        t.to(device)
        return t

    kf = KFold(n_splits=num_folds, random_state=0, shuffle=True)
    result = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        if train_verbose:
            print(f"\n<Fold {fold}>")

        train_loader = get_loader(dataset, sampler=SubsetRandomSampler(train_idx))
        val_loader = get_loader(dataset, sampler=SubsetRandomSampler(val_idx))

        fitter = initialize_trainer(fold)
        train_result, test_result = fitter.fit(train_loader, val_loader, split_result=True)
        result[fold] = dict(
            train_result=train_result, test_result=test_result,
            best_loss=fitter.best_loss, early_stopping=test_result.index(min(test_result)) + 1
        )
        # Plot Learning Curve to check over-fitting
        visualize_learning(
            train_result, test_result,
            title=f"Fold {fold} Learning Curve", figsize=(12, 12),
            filename=f"output_train_fold_{fold}.png", show=False
        )

    # todo
    #  (KFold 5개 --> early stopping 5개의 평균) --> random state 여러개로 평균  (평균 epoch 분포)
    early_stopping_epochs = []
    repeat = 100

    best_fold = 0
    best_loss = float('inf')
    for fold, fold_result in result.items():
        if fold_result['best_loss'] < best_loss:
            best_fold = fold
            best_loss = fold_result['best_loss']

    checkpoint = glob.glob(os.path.join(checkpoint_dir, f'fold_{best_fold}', 'best_checkpoint_epoch_*.pt'))[-1]
    model_state_dict = torch.load(checkpoint)['model']
    torch.save(model_state_dict, os.path.join(checkpoint_dir, "best_model_weight.pt"))

    net = get_model()
    net.load_state_dict(model_state_dict)
    net.eval().to(device)

    with torch.no_grad():
        prediction = []
        label = []
        mse = mae = 0.

        for inputs, labels in get_loader(dataset, train=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            mse += f.mse_loss(outputs, labels).item() * labels.shape[0]
            mae += f.l1_loss(outputs, labels).item() * labels.shape[0]
            prediction.append(outputs)
            label.append(labels)

        label = torch.cat(label, dim=0).view(-1, 1)
        prediction = torch.cat(prediction, dim=0).view(-1, 1)

    sc_val = dataset.y_proc.scaler_
    label = sc_val.inverse_transform(label.numpy())
    prediction = sc_val.inverse_transform(prediction.numpy())

    r2 = r2_score(label, prediction)
    mse /= len(dataset)
    mae /= len(dataset)

    file_output(f'MSE: {mse:.4f}\tMAE: {mae:.4f}\tTrain R2 score: {r2:.4f}')

    visualize_regression(
        label, prediction, mse_score=mse, mae_score=mae, r2_score=r2,
        plot_max=200, plot_min=0, vmax=40., vmin=0., alpha=0.5, figsize=(15, 12),
        xlabel='Value', ylabel='Prediction', title='Year of Survival Regression',
        filename=f"output_train_whole.png", show=False
    )


if __name__ == '__main__':
    train()
