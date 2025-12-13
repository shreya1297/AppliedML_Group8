def build_mcq_sequence(vocab, context, question, option, max_length):
    # [CLS] context [SEP] question option [SEP]
    ids = [vocab.cls_id]
    ids += vocab.encode_text(context)
    ids += [vocab.sep_id]
    ids += vocab.encode_text(question) + vocab.encode_text(option)
    ids += [vocab.sep_id]

    ids = ids[:max_length]
    attn = [1] * len(ids)

    # pad
    pad_len = max_length - len(ids)
    if pad_len > 0:
        ids += [vocab.pad_id] * pad_len
        attn += [0] * pad_len

    return ids, attn

def preprocess_mc_batch_scratch(batch, vocab, max_length=128):
    """
    Produces:
      input_ids: [B,4,L]
      attention_mask: [B,4,L]
      labels (optional): [B]
    """
    B = len(batch["context"])
    input_ids = []
    attention_mask = []

    for i in range(B):
        ctx = batch["context"][i]
        q = batch["question"][i]
        opts = batch["answers"][i]  # already a list of 4 strings
        ids4, attn4 = [], []
        for opt in opts:
            ids, attn = build_mcq_sequence(vocab, ctx, q, opt, max_length)
            ids4.append(ids)
            attn4.append(attn)
        input_ids.append(ids4)
        attention_mask.append(attn4)

    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "label" in batch:
        out["labels"] = batch["label"]
    return out
