from transformers import GPT2Tokenizer
import sys
import tracemalloc

# Start tracking memory allocation
tracemalloc.start()

# Load the GPT-2 tokenizer
tokenizers = []
for i in range(10):
    tokenizers.append(GPT2Tokenizer.from_pretrained("gpt2"))    # one model: approx. 23MB


# Measure memory usage
current, peak = tracemalloc.get_traced_memory()
memory_usage_bytes = current

# Stop tracking memory
tracemalloc.stop()

memory_usage_mb = memory_usage_bytes / (1024 ** 2)  # Convert bytes to megabytes
print(memory_usage_mb)
