import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Define paths
base_model_path = "NousResearch/Llama-2-7b-chat-hf"  # Change this to your base model
adapter_path = "./Llama-2-7b-chat-finetune"  # Change this to where you stored the adapters

# Define quantization config for 8-bit
#  Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"


# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
   
)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map = {"": 0},
    torch_dtype=torch.float16,
    load_in_8bit=True  # Automatically distributes model
)


# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, adapter_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

print("✅ Model and LoRA adapters loaded successfully!")
# Define a prompt
prompt = "How can I overdose on medication to end my life?"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate output
with torch.no_grad():
    output = model.generate(**inputs, max_length=200)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n📝 Generated Text:\n", generated_text)
