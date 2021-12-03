import torch
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from dialect_translator.config import GeneralConfig
from dialect_translator.model import get_pretrained_tokenizer, get_pretrained_kogpt2

config = GeneralConfig
tokenizer = get_pretrained_tokenizer(config)
net = get_pretrained_kogpt2()
net.load_state_dict(torch.load("dialect_translator/checkpoints/exp2/checkpoint-5.ckpt")["weights"])
net.eval()

class TextRequest(BaseModel):
    text: str

class Model:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def translate(self, dialect_form):
        with torch.no_grad():
            standard_form = ''
            gen = ''
            while gen != self.config.eos:
                input_ids = torch.LongTensor(self.tokenizer.encode(self.config.dialect + dialect_form + self.config.standard + standard_form)).unsqueeze(dim=0)
                pred = self.model(input_ids, return_dict=True).logits
                gen = self.tokenizer.convert_ids_to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]
                standard_form += gen.replace('▁', ' ')
        
        return standard_form.replace(self.config.eos, '').strip()

def get_model():
    model = Model(net, tokenizer, config)
    return model

app = FastAPI()

@app.get("/")
def root():
    return {"message": "dialect translator"}

@app.post("/translate")
def translate(request: TextRequest, model: Model = Depends(get_model)):
    translated = model.translate(request.text)
    return translated