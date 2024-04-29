from data.dataset_dpsr import DatasetDPSR

if __name__ == '__main__':
    from torch.utils.data import DataLoader


    options = {
        'n_channels': 3,
        'scale': 4,
        'H_size': 128,
        'sigma': [10, 50],
        'dataroot_H': '/home/ozkan/works/diff-smoe/dataset/STARE/train_0',
        'dataroot_L': None,
        'phw': 3,
        'stride': 1,
        'phase': 'train',
        'sigma_test': 0,
    }

   

    dataset = DatasetDPSR(options)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Example training loop
    for epoch in range(5):  # number of epochs
        for data in dataloader:
            L, L_p, H = data['L'], data['L_p'], data['H']