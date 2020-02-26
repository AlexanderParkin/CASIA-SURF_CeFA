import os
import torch


class State(object):
    """
    A class used to represent a state of model

    ...

    Attributes
    ----------
    root : Model
        Training model object.
    config : argparse.Namespace
        Training process config
    state : dict
        State dictionary to load and continue the training process.
        State includes epoch number, wrapper state dict, lr_scheduler, optimizer params etc.

    Methods
    -------
    create()
        Create self.state dictionary
    save()
        Save state
    save_checkpoint(filename)
        Save checkpoint to *filename*
    load_checkpoint()
        load checkpoint from .pth and set self.root attributes
    """
    def __init__(self, root):
        self.root = root
        self.config = self.root.config
        self.root_dir = self.config.checkpoint_config.out_path
        self.state = None

    def create(self):
        # Params to be saved in checkpoint
        self.state = {
            'epoch': self.root.epoch,
            'state_dict': self.root.wrapper.state_dict(),
            'lr_scheduler': self.root.lr_scheduler.state_dict(),
            'optimizer': self.root.optimizer.state_dict(),
        }

    def save(self):
        if self.config.checkpoint_config.save_frequency == 0:
            self.save_checkpoint('checkpoint.pth')
        else:
            if self.root.epoch % self.config.checkpoint_config.save_frequency == 0:
                self.save_checkpoint('model_{}.pth'.format(self.root.epoch))

    def save_checkpoint(self, filename):  # Save model to task_name/checkpoints/filename.pth
        fin_path = os.path.join(self.root_dir, 'checkpoints', filename)
        torch.save(self.state, fin_path)

    def load_checkpoint(self):  # Load current checkpoint if exists
        fin_path = os.path.join(self.root_dir, 'checkpoints', self.root.config.resume)
        if os.path.isfile(fin_path):
            print(">>>> loading checkpoint '{}'".format(fin_path))
            checkpoint = torch.load(fin_path, map_location='cpu')
            self.root.epoch = checkpoint['epoch'] + 1
            self.root.model.load_state_dict(checkpoint['state_dict'])
            self.root.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.root.optimizer.load_state_dict(checkpoint['optimizer'])

            print(">>>> loaded checkpoint '{}' (epoch {})".format(self.root.config.resume, checkpoint['epoch']))
        else:
            print(">>>> no checkpoint found at '{}'".format(self.root.config.resume))

