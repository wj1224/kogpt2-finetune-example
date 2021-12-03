import os
import copy

import torch
from tqdm import tqdm
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from dataset import create_dataloader
from utils import select_device, WandbLogger

class Trainer:
    def __init__(self, model, tokenizer, general_config, trainer_config, checkpoint, args):
        self.model = model
        self.tokenizer = tokenizer
        self.general_config = general_config
        self.trainer_config = trainer_config
        self.checkpoint = checkpoint
        self.args = args

        self.device = select_device(trainer_config.device)
        self.model = torch.nn.DataParallel(self.model).to(self.device) if self.trainer_config.dp else self.model.to(self.device)

        self.logger = WandbLogger(general_config, trainer_config, checkpoint["wandb_id"] if checkpoint is not None else None, args) if args.wandb else None
        self.samples = {}

    def save_checkpoint(self, epoch, optimizer, scheduler, scaler, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        model = self.model.module if hasattr(self.model, 'module') else self.model

        save_path = os.path.join(save_dir, f"checkpoint-{epoch}.ckpt")
        state = {'epoch': epoch,
                 'weights': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'scaler': scaler.state_dict(),
                 'wandb_id': self.logger.get_wandb_id() if self.logger is not None else None}
        torch.save(state, save_path)
        print(f"Checkpoint is saved at {save_path}")

    def load_checkpoint(self, optimizer, lr_scheduler, scaler):
        checkpoint = self.checkpoint

        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler["scheduler"].load_state_dict(checkpoint["scheduler"])
        if scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])

    def get_optimizers(self, length, config):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=config.lr, correct_bias=False)

        num_train_steps = length * config.epochs
        num_warmup_steps = int(num_train_steps * config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return optimizer, lr_scheduler

    def train(self, epochs):
        model, tokenizer = self.model, self.tokenizer
        general_config, trainer_config = self.general_config, self.trainer_config

        train_dataloader = create_dataloader(trainer_config.batch_size,
            trainer_config.num_workers,
            trainer_config.shuffle,
            tokenizer,
            general_config,
            train=True)

        val_dataloader = create_dataloader(trainer_config.batch_size,
            trainer_config.num_workers,
            shuffle=False,
            tokenizer=tokenizer,
            config=general_config,
            train=False)

        for idx, (token_ids, _) in enumerate(val_dataloader):
            if idx > 3:
                break
            self.samples[self.tokenizer.decode(token_ids[0], skip_special_tokens=False).split(general_config.dialect)[1].split(general_config.standard)[0][1:]] = None


        optimizer, lr_scheduler = self.get_optimizers(len(train_dataloader), trainer_config)
        scaler = torch.cuda.amp.GradScaler(enabled=trainer_config.amp)
        start_epoch = 0
        if self.checkpoint is not None:
            self.load_checkpoint(optimizer, lr_scheduler, scaler)
            start_epoch = self.checkpoint["epoch"]

        def train_step(epoch):
            model.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            loss_avg = 0.
            for idx, (token_ids, mask) in pbar:
                token_ids = token_ids.to(self.device)
                mask = mask.to(self.device)

                with torch.cuda.amp.autocast(enabled=trainer_config.amp):
                    outputs = model(token_ids, attention_mask=mask, labels=token_ids)
                    loss = outputs[0]
                    if trainer_config.dp is True:
                        loss = loss.mean()
                    loss_avg += loss.detach().item()

                optimizer.zero_grad()
                if trainer_config.amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.grad_norm_clip)
                    optimizer.step()
                lr_scheduler['scheduler'].step()
                
                pbar.set_description(f"epoch {epoch + 1} iter {idx}: train loss {loss_avg / (idx + 1):.5f}. lr {lr_scheduler['scheduler'].get_last_lr()[0]:e}")
            
            return loss_avg / len(train_dataloader)

        @torch.no_grad()
        def evaluation(epoch):
            model.eval()
            total_loss = 0.

            for _, (token_ids, mask) in enumerate(val_dataloader):
                token_ids = token_ids.to(self.device)
                mask = mask.to(self.device)

                outputs = model(token_ids, labels=token_ids)
                loss_avg = outputs[0]
                if trainer_config.dp is True:
                    loss_avg = loss_avg.mean()
                total_loss += loss_avg.item()
            total_loss /= len(val_dataloader)
            print(f"epoch {epoch + 1} validation loss {total_loss}")

            return total_loss

        for e in range(start_epoch, epochs):
            loss = train_step(e)
            val_loss = evaluation(e)
            if self.logger:
                values = (e + 1, loss, val_loss)
                translated = self.sample_translate(general_config)
                self.logger.end_epoch(values, translated, e + 1 == epochs)
            self.save_checkpoint(e + 1,
                                 optimizer,
                                 lr_scheduler["scheduler"],
                                 scaler,
                                 os.path.join(trainer_config.save_dir, self.args.name))

    def sample_translate(self, config):
        model_cp = copy.deepcopy(self.model.module if hasattr(self.model, 'module') else self.model)
        model_cp = model_cp.to(torch.device("cpu"))
        model_cp.eval()
        tokenizer = self.tokenizer
        samples = self.samples

        with torch.no_grad():
            for input_texts, _ in samples.items():
                standard_form = ''
                gen = ''
                while gen != config.eos:
                    input_ids = torch.LongTensor(tokenizer.encode(config.dialect + input_texts + config.standard + standard_form, return_tensors='pt')).unsqueeze(dim=0)
                    pred = model_cp(input_ids, return_dict=True).logits
                    gen = tokenizer.convert_ids_to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    standard_form += gen.replace('▁', ' ')
                    samples[input_texts] = standard_form.replace(config.eos, '').strip()

        return samples