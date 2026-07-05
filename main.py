from src.model.gpt import GPT2Model
import tiktoken
import torch

cfg = {
        "vocab_size":50257,
        "context_length":1024,
        "emb_dim":768,
        "n_heads":12,
        "n_layers":12,
        "drop_rate":0.1,
        "qkv_bias":False
}

def generate_text(model,idx,max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        probas = torch.softmax(logits,dim=-1)
        id_next = torch.argmax(probas,dim=-1,keepdim=True)
        idx = torch.cat((idx,id_next),dim=1)

    return idx

tokenizer = tiktoken.get_encoding("gpt2")
model = GPT2Model(cfg)

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded tensor shape: ",encoded_tensor.shape)

out = generate_text(
    model=model,
    idx = encoded_tensor,
    max_new_tokens=10,
    context_size=cfg["context_length"],
)

print(tokenizer.decode(out.squeeze(0).tolist()))