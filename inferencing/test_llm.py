from mlx_lm import load, generate
from inference01_config import LLM_MODEL, SYSTEM_PROMPT

print("Loading model")
model, tokenizer = load(LLM_MODEL)
print("Model loaded!\n")

test_cases = [
    "assalomu_alaykum, ism, nima",
    "men, maktab, borish",
    "kitob, qayerda",
    "men, ovqat, tayyorlash, xohlash"
]

for test in test_cases:
  prompt = f"{SYSTEM_PROMPT}\n\nInput: {test}\nOutput:"

  print(f"Input signs: {test}")

  response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=50,
    verbose=False
  )
  print(f"LLM output: {response}")
  print("-"*50)