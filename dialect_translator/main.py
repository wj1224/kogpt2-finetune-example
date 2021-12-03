import argparse

import torch
from config import GeneralConfig
from model import get_pretrained_tokenizer, get_pretrained_kogpt2

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="training mode")
parser.add_argument("--freeze", action="store_true", help="layer freeze on training mode")
parser.add_argument("--seed", type=int, default=0, help="set seed for reproducible")
parser.add_argument("--checkpoint", type=str, default=None, help="trained checkpoint")
parser.add_argument("--wandb", action="store_true", help="use wandb logging")
parser.add_argument("--project", type=str, default="kogpt2-train", help="set wandb project")
parser.add_argument("--entity", type=str, default=None, help="set wandb entity")
parser.add_argument("--name", type=str, default="exp", help="set experiment name")
args = parser.parse_args()

if __name__ == "__main__":
    general_config = GeneralConfig
    checkpoint = None

    model = get_pretrained_kogpt2()
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["weights"])
    tokenizer = get_pretrained_tokenizer(general_config)

    if args.train:
        from trainer import Trainer
        from config import TrainerConfig

        if args.freeze:
            from utils import freeze_layers
            freeze_layers(model)

        trainer_config = TrainerConfig
        trainer = Trainer(model, tokenizer, general_config, trainer_config, checkpoint, args)
        trainer.train(trainer_config.epochs)

    else:
        model.eval()

        with torch.no_grad():
            while True:
                dialect_form = input('Dialect form: ').strip()
                standard_form = ''
                gen = ''
                while gen != general_config.eos:
                    input_ids = torch.LongTensor(tokenizer.encode(general_config.dialect + dialect_form + general_config.standard + standard_form)).unsqueeze(dim=0)
                    pred = model(input_ids, return_dict=True).logits
                    gen = tokenizer.convert_ids_to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    standard_form += gen.replace('▁', ' ')
                print("Standard from: {}".format(standard_form.replace(general_config.eos, '').strip()))