from src.runner.trainers import BaseTrainer
import torch.nn.functional as F

class Trainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_step(self, batch):
        input_img, target = batch['img'].to(self.device), batch['label'].to(self.device)
        output = self.net(input_img)
        loss = self.loss_fns.cross_entropy_loss(output, target)
        accuracy = self.metric_fns.Accuracy(F.softmax(output, dim=1), target)
        return {
            'outputs': output,
            'loss': loss,
            'metrics': {
                'Accuracy': accuracy
            }
        }

    def _valid_step(self, batch):
        input_img, target = batch['img'].to(self.device), batch['label'].to(self.device)
        output = self.net(input_img)
        loss = self.loss_fns.cross_entropy_loss(output, target)
        accuracy = self.metric_fns.Accuracy(F.softmax(output, dim=1), target)
        return {
            'outputs': output,
            'loss': loss,
            'metrics': {
                'Accuracy': accuracy
            }
        }