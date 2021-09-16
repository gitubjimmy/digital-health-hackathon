import torch


@torch.no_grad()
def test():

    import os
    import pandas as pd
    import torch

    from models import get_model

    net = get_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(os.path.join('checkpoint', "best_model_weight.pt")))
    net.eval().to(device)

    # t1_input: treatment == 1
    # t2_input: treatment == 0

    def get_tau_input(num_gene, has_gene, treatment):

        assert isinstance(num_gene, int) and 1 <= num_gene <= 300
        assert has_gene in (0, 1)  # allowed bool (True, False): bool is subclass of integer
        assert treatment in (0, 1)  # allowed bool (True, False): bool is subclass of integer

        import torch

        try:
            get_tau_input.cache
        except AttributeError:
            from data_prep_utils import get_processed_data
            get_tau_input.cache = get_processed_data().samples.mean(axis=0).drop('time')

        df = get_tau_input.cache.copy()
        df['Treatment'] = int(treatment)
        df['G{}'.format(num_gene)] = int(has_gene)
        return torch.from_numpy(df.to_numpy())

    t1_pos_inputs = []
    t1_neg_inputs = []
    t2_pos_inputs = []
    t2_neg_inputs = []

    for g in range(1, 300 + 1):
        t1_pos_inputs.append(get_tau_input(g, has_gene=1, treatment=1))
        t1_neg_inputs.append(get_tau_input(g, has_gene=0, treatment=1))
        t2_pos_inputs.append(get_tau_input(g, has_gene=1, treatment=0))
        t2_neg_inputs.append(get_tau_input(g, has_gene=0, treatment=0))

    t1_pos_tensor = torch.stack(t1_pos_inputs).to(torch.float32).to(device)
    t1_neg_tensor = torch.stack(t1_neg_inputs).to(torch.float32).to(device)
    t2_pos_tensor = torch.stack(t2_pos_inputs).to(torch.float32).to(device)
    t2_neg_tensor = torch.stack(t2_neg_inputs).to(torch.float32).to(device)

    tau1_score = torch.abs(net(t1_pos_tensor) - net(t1_neg_tensor)).squeeze()
    tau2_score = torch.abs(net(t2_pos_tensor) - net(t2_neg_tensor)).squeeze()
    tau_score = torch.abs(tau1_score - tau2_score)

    arr = torch.stack([tau1_score, tau2_score, tau_score]).transpose(0, 1).cpu().numpy()
    df = pd.DataFrame(
        arr,
        index=tuple(map(lambda key: "G" + str(key), range(1, arr.shape[0] + 1))),
        columns=("tau1", "tau2", "abs(t1-t2)")
    )

    print("# Tau Score from MLP  \n")
    print()
    print("## Sort by tau1\n")
    print(df.sort_values(by="tau1", ascending=False).to_markdown())
    print()
    print("## Sort by tau2\n")
    print(df.sort_values(by="tau2", ascending=False).to_markdown())
    print()
    print("## Sort by abs(tau1-tau2)\n")
    print(df.sort_values(by="abs(t1-t2)", ascending=False).to_markdown())
    print()

    return df


if __name__ == '__main__':
    test()
