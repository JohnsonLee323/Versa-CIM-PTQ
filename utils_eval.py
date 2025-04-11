import sys
import torch

import numpy as np

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# imagenet validation---------------------------------------------------------------------------------------------------

def test_imagenet(net, val_loader, max_iteration=None, description=None):
    pos = 0
    tot = 0
    i = 0
    max_iteration = len(val_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q = tqdm(val_loader, desc=description)
        for inp, target in q:
            i += 1
            inp = inp.cuda()
            target = target.cuda()
            out = net(inp)
            pos_num = torch.sum(out.argmax(1) == target).item()
            pos += pos_num
            tot += inp.size(0)
            q.set_postfix({"acc": pos / tot})
            if i >= max_iteration:
                break
    return (pos / tot) * 100

# imagenet validation---------------------------------------------------------------------------------------------------


# lambada validation----------------------------------------------------------------------------------------------------

def test_lambada(config, model, tokenizer, val_loader, max_iteration=None, description=None):
    errors = 0
    total = 0
    max_iteration = len(val_loader) if max_iteration is None else max_iteration
    for i, batch in enumerate(tqdm(val_loader, desc=description)):
        errors_batch, total_batch = score_batch(config, model, tokenizer, batch)
        errors += errors_batch
        total += total_batch
        if i >= max_iteration:
            break
    return (1 - errors / total) * 100


def score_batch(config, model, tokenizer, batch):
    """Return number of last-word mismatches in a batch."""
    batch_encoded = []
    lengths = []
    fragments = []
    for line in batch:
        line = line.strip()
        line_encoded = tokenizer.encode(line)

        encoded_last_word = tokenizer.decode(line_encoded[-1:]).strip()
        actual_last_word = line.split()[-1].strip()

        if encoded_last_word != actual_last_word:
            fragments.append(True)
        else:
            fragments.append(False)
        batch_encoded.append(line_encoded)

    # array is ragged, so pad to turn into rectangular tensor
    max_len = max(len(encoded) for encoded in batch_encoded)

    batch_padded = []
    for encoded in batch_encoded:

        batch_padded.append(encoded + [0] * (max_len - len(encoded)))
        lengths.append(len(encoded))

    batch_padded = torch.tensor(batch_padded)
    batch_padded = batch_padded.to(device)

    with torch.no_grad():
        outputs = model(batch_padded)
        logits = outputs.logits

    errors = 0
    total = 0
    for i in range(config.DATA.BATCH_SIZE):
        # break on small last batch
        if i >= len(batch_padded):
            break

        last_idx = lengths[i] - 1
        observed = batch_encoded[i][last_idx]

        predicted = argmax(logits[i][last_idx - 1])

        total += 1
        errors += 0 if (observed == predicted) else 1

    return errors, total


def argmax(t):
    return int(torch.argmax(t).item())

# lambada validation----------------------------------------------------------------------------------------------------


# VAD-------------------------------------------------------------------------------------------------------------------

def test_vad(val_loader, model):
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, targets = data

            targets = targets.view(-1).cuda()  # 确保目标是一维张量
            output = model(inputs.cuda())

            # 计算预测类别并累计正确数
            _, preds = torch.max(output, 1)
            preds = preds.long().view(targets.size())
            n_correct += (preds == targets).sum().item()
            n_total += inputs.size(0)

    # 计算并返回准确率（百分比，保留两位小数）
    accuracy = 100.0 * n_correct / n_total
    return accuracy

# VAD-------------------------------------------------------------------------------------------------------------------


# WikiText2-------------------------------------------------------------------------------------------------------------

def test_ppl(config, model, tokenizer, val_loader):
    encoding = tokenizer("\n\n".join(val_loader["text"]), return_tensors="pt")
    seq_len = encoding.input_ids.shape[1]
    C = 1024 ## context length
    log_prob = []
    for begin_loc in tqdm(range(0, seq_len, C)):
        end_loc = min(begin_loc + C, seq_len)
        input_ids = encoding.input_ids[:, begin_loc: end_loc].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            logits = logits[:, :-1]
            labels = input_ids[:, 1:]
            probs = torch.softmax(logits, dim=-1)
            probs = probs.squeeze(0)
            labels = labels.squeeze(0)
            target_probs = torch.gather(probs, 1, labels.unsqueeze(1))
            log_prob.extend(target_probs.log2().cpu().numpy().tolist())
        if end_loc == seq_len:
            break
    ce = - np.sum(log_prob) / len(log_prob)
    ppl = 2 ** ce
    return ppl

# WikiText2-------------------------------------------------------------------------------------------------------------
