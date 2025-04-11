from torchvision.models import resnet50
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'resnet':
        model = resnet50(pretrained=config.MODEL.PRETRAINED)
        tokenizer = None
    elif model_type == 'gpt2':
        if config.MODEL.PRETRAINED:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
    else:
        raise NotImplementedError(f"Unknown model: {model_type}")

    return model, tokenizer
