from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

index = 0
resids = {}


def get_activation(idx):
    def hook(module, input, output):
        resids[idx] = output
    return hook


for layer in model.model.layers:
    h = layer.self_attn.register_forward_hook(get_activation(index))
    index += 1

prompt = "Give me a short introduction to large language model."
model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
model.generate(
    **model_inputs,
    max_new_tokens=1
)
print(resids)
