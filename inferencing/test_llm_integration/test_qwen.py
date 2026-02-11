from mlx_lm import load, generate
# from inference01_config import LLM_MODEL, SYSTEM_PROMPT

LLM_MODEL="mlx-community/Qwen2.5-1.5B-Instruct-4bit"
SYSTEM_PROMPT = """Siz berilgan so'zlardan to'g'ri va mantiqiy jumla tuzasiz.

Qoidalar:
- So'zlarni grammatik jihatdan to'g'ri bog'lang
- Mantiqli va tushunarli jumla yarating
- Faqat yakuniy jumlani yozing, boshqa hech narsa qo'shmang

Misollar:

So'zlar: men, maktab, borish
Jumla: Men maktabga boraman.

So'zlar: kitob, qayerda, kutubxona
Jumla: Kitob kutubxonada.

So'zlar: ovqat, tayyorlash, oshxona
Jumla: Ovqatni oshxonada tayyorlayman.

So'zlar: iltimos, yordam, kerak
Jumla: Iltimos, menga yordam kerak.

Endi jumla tuzing:"""


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