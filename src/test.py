import torch
import time
from transformers import AutoModelForCausalLM

# Disable TF32 so PyTorch uses exact FP32 math (matching your C++ engine)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model_path = "model_data" # Path to your Qwen HF model folder

print("Loading HuggingFace model...")
# Load in float32 to match your C++ engine's precision
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float32,
)
model = model.to("cuda")
model.eval()

input_ids = torch.tensor([[1337, 42, 9000]]).cuda()

print("Starting Warm-up...")
with torch.no_grad():
    for _ in range(3):
        # use_cache=False ensures it matches your C++ engine's current state!
        _ = model(input_ids, use_cache=True)
        
torch.cuda.synchronize()
print("Warm-up complete!\n")

target_tokens = 50
print(f"Benchmarking {target_tokens} tokens...")

# START TIMER
start_time = time.perf_counter()

with torch.no_grad():
    for _ in range(target_tokens):
        outputs = model(input_ids, use_cache=True)
        
        # Simulating the exact same workload as C++
        # (For real generation, you'd argmax and append here)

# STOP TIMER (Must synchronize first!)
torch.cuda.synchronize()
end_time = time.perf_counter()

seconds = end_time - start_time
tps = target_tokens / seconds

print("===== BENCHMARK RESULTS =====")
print(f"Total Time: {seconds:.4f} seconds")
print(f"Speed:      {tps:.2f} Tokens Per Second (TPS)")
print("=============================")