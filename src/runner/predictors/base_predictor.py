import logging
import torch
from tqdm import tqdm
import numpy as np

from src.runner.utils import EpochLog

LOGGER = logging.getLogger(__name__.split('.')[-1])


class BasePredictor:
    """The base class for all predictors.
    Args:
        saved_dir (Path): The root directory of the saved data.
        device (torch.device): The device.
        test_dataloader (Dataloader): The testing dataloader.
        net (BaseNet): The network architecture.
        loss_fns (LossFns): The loss functions.
        loss_weights (LossWeights): The corresponding weights of loss functions.
        metric_fns (MetricFns): The metric functions.
    """

    def __init__(self, saved_dir, device, test_dataloader,
                 net, metric_fns):
        self.saved_dir = saved_dir
        self.device = device
        self.test_dataloader = test_dataloader
        self.net = net.to(device)
        # self.loss_fns = loss_fns
        # self.loss_weights = loss_weights
        self.metric_fns = metric_fns

    def predict(self):
        """The testing process.
        """
        self.net.eval()
        dataloader = self.test_dataloader
        pbar = tqdm(dataloader, desc='test', ascii=True)

        epoch_log = EpochLog()
        ans = open('./effnet_b3fc_ans.csv', 'w')
        ans.write('filename,category\n')

        logits_store = []
        for i, batch in enumerate(pbar):
            with torch.no_grad():
                input_img = batch['img'].to(self.device)
                files = batch['filename']
                output = self.net(input_img)
                pred = output.argmax(dim=1, keepdim=False)
                for f, p, logit in zip(files, pred, output):
                    ans.write('{},{}\n'.format(f, str(p.item()).zfill(2)))
                    logits_store.append(logit)
                # test_dict = self._test_step(batch)
                # loss = test_dict['loss']
                # losses = test_dict.get('losses')
                # metrics = test_dict.get('metrics')

            if (i + 1) == len(dataloader) and not dataloader.drop_last:
                batch_size = len(dataloader.dataset) % dataloader.batch_size
            else:
                batch_size = dataloader.batch_size
        
        logits_store = np.array(logits_store)
        print(logits_store.shape)
        np.save('effnet_b3fc_aug.npy', logits_store)
        #     epoch_log.update(batch_size, loss, losses, metrics)

        #     pbar.set_postfix(**epoch_log.on_step_end_log)
        # test_log = epoch_log.on_epoch_end_log
        # LOGGER.info(f'Test log: {test_log}.')

    def _test_step(self, batch):
        """The user-defined testing logic.
        Args:
            batch (dict or sequence): A batch of the data.

        Returns:
            test_dict (dict): The computed results.
                test_dict['loss'] (torch.Tensor)
                test_dict['losses'] (dict, optional)
                test_dict['metrics'] (dict, optional)
        """
        # raise NotImplementedError
        pass

    def load(self, path):
        """Load the model checkpoint.
        Args:
            path (Path): The path to load the model checkpoint.
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.net.load_state_dict(checkpoint['net'])
