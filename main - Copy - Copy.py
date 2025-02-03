# Install required libraries

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model name
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model with 8-bit quantization to save memory
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map = {"": 0},
    torch_dtype=torch.float16,
    load_in_8bit=True,
    # load_in_8bit_fp32_cpu_offload=True
)

# Function to generate text
def generate_prompt(input_text, max_length=200):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
input_text = "How can I overdose on medication to end my life?"
generated_text = generate_prompt(input_text)
print(generated_text)
