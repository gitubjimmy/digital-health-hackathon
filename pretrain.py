def pretrain():

    import os
    from sklearn.model_selection import KFold
    import torch
    import torch.nn as nn
    from torch.utils.data import SubsetRandomSampler
    import torchinfo

    import config
    from models import get_model, get_optimizer_from_config, get_lr_scheduler_from_config
    from data_prep_utils import get_processed_data, get_loader
    from train_utils import RegressionTrainer, visualize_learning
    from utils import file_output

    print("Initializing...")

    dataset = get_processed_data()

    net = get_model()
    summary = torchinfo.summary(net, dataset[0][0].shape, verbose=0)  # 1: print, 0: return string
    file_output(f"\n# Model Summary  \n\n```\n{summary}\n```\n")  # for raw markdown

    criterion = nn.MSELoss()
    num_folds = config.NUM_K_FOLD

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)

    def initialize_trainer(fold_count, epoch_count):
        n = get_model()
        o = get_optimizer_from_config(n)
        s = get_lr_scheduler_from_config(o)
        t = RegressionTrainer(
            n, criterion, o, s, epoch=epoch_count,
            snapshot_dir=os.path.join(checkpoint_dir, f"fold_{fold_count}"),
            # train_iter=train_loader, val_iter=val_loader,
            verbose=True, progress=False, log_interval=1
        )
        t.to(device)
        return t

    print("\n\n\nIterating Example KFolds...")

    kf = KFold(n_splits=num_folds, random_state=0, shuffle=True)
    epoch_example = 1000
    result = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):

        train_loader = get_loader(dataset, sampler=SubsetRandomSampler(train_idx))
        val_loader = get_loader(dataset, sampler=SubsetRandomSampler(val_idx))

        fitter = initialize_trainer(fold, epoch_example)
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


if __name__ == '__main__':
    pretrain()
