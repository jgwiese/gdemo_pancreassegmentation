import sys
import os
from os.path import join
import torch
import random
import numpy as np

project_dir = './'
data_dir = join(project_dir, 'data')
src_dir = join(project_dir, 'src')
model_dir = join(project_dir, 'model')
output_dir = join(project_dir, 'output')
sys.path.append(src_dir)

import Autoencoder
import Data
import Solver

def loss_function(output, x):
    recon_x, mu, logVar = output
    batchSize = mu.shape[0]
    rl = (recon_x - x).pow(2).sum() / batchSize
    kld = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp()) / batchSize
    return rl + kld

if __name__ == '__main__':
    # data
    dataset = Data.PancreasDataset(data_dir=data_dir, lod=6)
    split, samples = 0.9, len(dataset)
    dataset_train, dataset_validate = torch.utils.data.random_split(dataset, [int(split * samples), samples - int(split * samples)])

    # model, optimizer
    model = Autoencoder.VariationalAutoencoder(imageShape=(1, 2 ** 7, 2 ** 7), firstFilterCount=16, act=torch.nn.functional.elu, layerwise=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Solver
    loadWeights = True
    solver = Solver.Solver(model=model, modelDir=model_dir, loadWeights=loadWeights, optimizer=optimizer, criterions=[loss_function, torch.nn.BCELoss()], iouThreshold=0.2)

    # routine: train on lod=6, hence (64, 64)
    epochs = 101
    lod = 6
    batchSize = 1024
    iterations = 1
        
    dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batchSize, shuffle=True, num_workers=2)
    dataloader_validate = torch.utils.data.DataLoader(dataset=dataset_validate, batch_size=batchSize, shuffle=True, num_workers=2)

    for i in range(iterations):
        solver.train(dataloader=dataloader_train, epochs=epochs, lod=lod)
        solver.evaluate(dataloader=dataloader_validate, lod=lod)
        solver.saveReconstructions(dataloader=dataloader_validate, lod=lod, count=10, output_dir=output_dir)
        #solver.saveCheckpoint()