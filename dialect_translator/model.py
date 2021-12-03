from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

def get_pretrained_tokenizer(config):
    return PreTrainedTokenizerFast.from_pretrained(
        "skt/kogpt2-base-v2",
        bos_token=config.bos,
        eos_token=config.eos,
        unk_token=config.unk,
        pad_token=config.pad,
        mask_token=config.mask
        )

def get_pretrained_kogpt2():
    return GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
