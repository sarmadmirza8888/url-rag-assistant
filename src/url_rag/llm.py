"""Local Mistral 7B Instruct (4-bit) text generation. Needs a CUDA GPU. Loads lazily."""
import os

MODEL_PATH = os.getenv(
    "MODEL_PATH", "/kaggle/input/datasets/<user>/mistral-7b-instruct-4bit"
)
_tokenizer = None
_llm = None


def load_model():
    global _tokenizer, _llm
    if _llm is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _llm = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, device_map="auto", torch_dtype="auto"
        )
    return _tokenizer, _llm


def generate(messages, max_new_tokens: int = 200) -> str:
    """Run a chat-formatted prompt through the local model and return the new text only."""
    import torch

    tokenizer, llm = load_model()
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)

    with torch.inference_mode():
        output = llm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens, not the prompt
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
