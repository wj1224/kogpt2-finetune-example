from logging import logProcesses
import os

import wandb
import torch
import torch.backends.cudnn as cudnn

def select_device(device):
    if device.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        return torch.device("cuda")

def set_reproducibility(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        cudnn.benchmark, cudnn.deterministic = False, True

def freeze_layers(model):
    """ freeze bottom 6 layers """
    for parameter in model.parameters():
        parameter.requires_grad = False

    for i, m in enumerate(model.transformer.h):
        if i >= 6:
            for parameter in m.parameters():
                parameter.requires_grad = True

    for parameter in model.transformer.ln_f.parameters():
        parameter.requires_grad = True

    for parameter in model.lm_head.parameters():
        parameter.requires_grad = True

class WandbLogger:
    def __init__(self, general_config, trainer_config, wandb_id, args):
        self.config = self.get_config_dict(general_config, trainer_config)

        self.wandb_run = wandb.init(
            config=self.config,
            resume="allow",
            project=args.project,
            entity=args.entity,
            name=args.name,
            job_type="train",
            id=wandb_id
            )

        self.text_table = wandb.Table(columns=["epoch", "val_loss", "dialect form", "standard form"])

    def get_wandb_id(self):
        return self.wandb_run.id

    def get_config_dict(self, general_config, trainer_config):
        config = dict()

        for k, v in vars(general_config).items():
            if not k.startswith("__"):
                config[k] = v
            else:
                pass
        for k, v in vars(trainer_config).items():
            if not k.startswith("__"):
                config[k] = v
            else:
                pass
        
        return config

    def log(self, epoch, logs, last_epoch=False):
        if last_epoch:
            logs["translated samples"] = self.text_table
        self.wandb_run.log(logs, step=epoch, commit=True)

    def log_values(self, values):
        logs = dict()
        logs["loss"] = values[1]
        logs["val_loss"] = values[2]
        
        return logs

    def _log_texts(self, values, texts):
        epoch, val_loss = values[0], values[2]
        for sample, output in texts.items():
            self.text_table.add_data(epoch, val_loss, sample, output)

    def end_epoch(self, values, texts, last_epoch=False):
        # values: (epoch, loss, val_loss)
        logs = self.log_values(values)
        self._log_texts(values, texts)
        self.log(values[0], logs, last_epoch=last_epoch)