import os
from os.path import join
import numpy as np
import torch
from matplotlib import pyplot as plt

class Solver():
    def __init__(self, model, modelDir, loadWeights, optimizer, criterions, iouThreshold):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.modelDir = modelDir
        if loadWeights:
            self.loadCheckpoint(join(self.modelDir, 'model-checkpoint.pt'))
        self.model.to(self.device)

        self.criterions = criterions
        self.iouThreshold = iouThreshold
    
    def loadCheckpoint(self, checkpointPath):
        checkpoint = torch.load(checkpointPath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    def saveCheckpoint(self):
        checkpointPath = join(self.modelDir, 'model-checkpoint.pt')
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpointPath)

    def formatScientific(self, number):
        if number < 0.001:
            return np.format_float_scientific(number, unique=False, precision=3)
        return '{:.04f}'.format(number)

    def train(self, dataloader, epochs, lod, printAt=10):
        for epoch in range(epochs):
            batchCounter = 0
            losses = np.array([0] * len(self.criterions), dtype=np.float32)
            for batch in dataloader:
                y = self.model.forward(batch['x'].to(self.device), lod)

                # Backpropagation
                self.model.zero_grad()
                for i, value in enumerate(zip(y, self.criterions, batch)):
                    output, criterion, key = value
                    loss = criterion(output, batch[key].to(self.device))
                    loss.backward()
                    losses[i] += loss
                self.optimizer.step()

                batchCounter += 1

            if ((epoch % printAt) == 0):
                losses = losses / batchCounter
                print("epoch {}, reconstruction/kld loss: {}, segmentation loss: {}".format(
                    epoch,
                    self.formatScientific(losses[0]),
                    self.formatScientific(losses[1])))

    def intersectionOverUnion(self, label, label_reconstruction):
        """Calculates the IoU metric and returns the result within (0, 1)."""
        intersection = ((label >= self.iouThreshold) & (label_reconstruction >= self.iouThreshold)) * 1.0
        union = ((label >= self.iouThreshold) | (label_reconstruction >= self.iouThreshold)) * 1.0
        iou = intersection.sum() / union.sum()
        return iou / label.shape[0]

    def evaluate(self, dataloader, lod):
        batchCounter = 0
        losses = np.array([0] * len(self.criterions), dtype=np.float32)
        iou = 0
        for batch in dataloader:
            y = self.model.forward(batch['x'].to(self.device), lod)

            # Loss calculation
            for i, value in enumerate(zip(y, self.criterions, batch)):
                output, criterion, key = value
                loss = criterion(output, batch[key].to(self.device))
                losses[i] += loss
            
            # Intersection over Union (IoU) measure:
            iou += self.intersectionOverUnion(
                label=batch['t'].to(self.device),
                label_reconstruction=y[1])
            batchCounter += 1
        
        losses /= batchCounter
        iou /= batchCounter
        print("evaluation, reconstruction loss: {}, segmentation loss: {}, IoU: {}".format(
            self.formatScientific(losses[0]),
            self.formatScientific(losses[1]),
            self.formatScientific(iou)))
    
    def reconstruct(self, dataloader, lod, count):
        """Can be used to reconstruct maximal 'count' samples from dataloader by model."""
        batch = next(iter(dataloader))
        count = min(count, len(batch['x']))
        x = batch['x'][:count].to(self.device)
        t = batch['t'][:count].to(self.device)
        y = self.model.forward(x, lod)
        x_reconstruction = y[0][0]
        x_segmentation = y[1]
        return x, t, x_reconstruction, x_segmentation

    def saveReconstructions(self, dataloader, lod, count, output_dir):
        """Reconstructs maximally 'count' samples and saves them to output_dir."""
        x_ins, t_ins, x_outs, t_outs = self.reconstruct(dataloader, lod, count)

        x_ins = x_ins.detach().cpu().numpy()
        t_ins = t_ins.detach().cpu().numpy()
        x_outs = x_outs.detach().cpu().numpy()
        t_outs = t_outs.detach().cpu().numpy()
        
        for i, value in enumerate(zip(x_ins, t_ins, x_outs, t_outs)):
            x_in, t_in, x_out, t_out = value

            x = np.stack([x_in] * 3, axis=0).squeeze().transpose((1, 2, 0))
            t = np.stack([t_in] * 3, axis=0).squeeze().transpose((1, 2, 0))
            mask = t[..., 0] > self.iouThreshold

            x[mask] = np.array([0, 1, 0.5]) * t[mask]
            plt.imsave(join(output_dir, 'res_{}_sample_{}_in.png'.format(2 ** lod, i)), x)

            x_r = np.stack([x_out] * 3, axis=0).squeeze().transpose((1, 2, 0))
            t_r = np.stack([t_out] * 3, axis=0).squeeze().transpose((1, 2, 0))
            mask = t_r[..., 0] > self.iouThreshold
            x_r[mask] = np.array([0, 1, 0.5]) * t[mask]
            plt.imsave(join(output_dir, 'res_{}_sample_{}_out.png'.format(2 ** lod, i)), x_r)