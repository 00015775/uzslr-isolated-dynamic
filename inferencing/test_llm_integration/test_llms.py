import csv
import time
from datetime import datetime
import time
import re
import subprocess
import sys


# pip install mlx-lm
try:
  from mlx_lm import load, generate
  MLX_AVAILABLE = True
except ImportError:
  MLX_AVAILABLE = False
  print("Warning: mlx-lm not installed")

# pip install groq
try:
  from groq import Groq
  GROQ_AVAILABLE = True
except ImportError:
  GROQ_AVAILABLE = False
  print("Warning: groq not installed")

# pip install transformers langid accelerate
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed")

# curl -fsSL https://ollama.com/install.sh | sh 
# pip install ollama
# ollama pull qwen2.5:1.5b
# ollama pull qwen2.5:3b
try:
   import ollama
   OLLAMA_AVAILABLE = True
except ImportError:
   OLLAMA_AVAILABLE = False
   print("Warning: ollama not installed")
   

# these test cases are based on the 50 signs that original model was trained on
TEST_CASES = [
    ["bola", "kitob", "o'qish"],
    ["men", "bozor", "borish"],
    ["ustoz", "savol", "berish"],
    ["do'st", "uy", "kelish"],
    ["ona", "ovqat", "tayyorlash"],
    ["talaba", "dars", "boshlash"],

    ["men", "ertalab", "maktab", "borish"],
    ["bola", "kecha", "park", "o'ynash"],
    ["ota", "mashina", "ish", "haydash"],
    ["biz", "do'kon", "non", "sotib_olish"],
    ["ustoz", "talaba", "topshiriq", "tushuntirish"],

    ["men", "bugun", "kutubxona", "kitob", "o'qish"],
    ["bola", "do'st", "stadion", "futbol", "o'ynash"],
    ["ona", "bozor", "sabzavot", "kechqurun", "pishirish"],
    ["talaba", "universitet", "imtihon", "ertalab", "topshirish"],
    ["biz", "tog'", "dam_olish", "rasm", "olish"]
]


TEST_CASES_3_WORDS = [
    ["bola", "kitob", "o'qish"],
    ["men", "bozor", "borish"],
    ["ustoz", "savol", "berish"],
    ["do'st", "uy", "kelish"],
    ["ona", "ovqat", "tayyorlash"],
    ["talaba", "dars", "boshlash"]
]

TEST_CASES_4_WORDS = [
    ["men", "ertalab", "maktab", "borish"],
    ["bola", "kecha", "park", "o'ynash"],
    ["ota", "mashina", "ish", "haydash"],
    ["biz", "do'kon", "non", "sotib_olish"],
    ["ustoz", "talaba", "topshiriq", "tushuntirish"]
]

TEST_CASES_5_WORDS = [
    ["men", "bugun", "kutubxona", "kitob", "o'qish"],
    ["bola", "do'st", "stadion", "futbol", "o'ynash"],
    ["ona", "bozor", "sabzavot", "kechqurun", "pishirish"],
    ["talaba", "universitet", "imtihon", "ertalab", "topshirish"],
    ["biz", "tog'", "dam_olish", "rasm", "olish"]
]


SYSTEM_PROMPT = """Siz berilgan so'zlardan to'g'ri va mantiqiy jumla tuzasiz.

Qoidalar:
- So'zlarni grammatik jihatdan to'g'ri bog'lang
- Mantiqli va tushunarli jumla yarating
- Faqat yakuniy jumlani yozing, boshqa hech narsa qo'shmang
- Faqat BIRTA sodda va tabiiy jumla yozing
- Qo‘shimcha gap yozmang
- Izoh bermang
- So‘zlarning barchasidan foydalaning

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

# tests qwen2.5-1.5B
def text_mlx_model(model_name, model_id, test_cases, temperature=0.7, max_tokens=50):
  """Test MLX local model"""
  print(f"\nLoading {model_name}...")
  model, tokenizer = load(model_id)
  print(f"{model_name} loaded!\n")

  results = []

  for idx, words in enumerate(test_cases, 1):
    words_str = ", ".join(words)
    prompt = f"{SYSTEM_PROMPT}\n\nSo'zlar: {words_str}\nJumla:"

    print(f"Test {idx}: {words_str}")

    start_time = time.time()
    response = generate(
      model,
      tokenizer,
      prompt=prompt,
      max_tokens=max_tokens,
      verbose=False
    )

    elapsed = time.time() - start_time

    # clean response 
    sentence = response.strip()
    if "Jumla:" in sentence:
        sentence = sentence.split("Jumla:")[-1].strip()
    if "So'zlar:" in sentence:
        sentence = sentence.split("So'zlar:")[0].strip()

    print(f"Output: {sentence}")
    print(f"Time: {elapsed:.2f}s\n")

    results.append({
      "model": model_name,
      "test_id": idx,
      "input_words": words_str,
      "output_sentence": sentence,
      "time_seconds": round(elapsed, 2)
    })
  
  return results



# tests llama-3.3-70b-versatile
def test_groq_model(model_name, model_id, test_cases, temperature=0.7, max_tokens=50):
  """Test Groq API model"""
  print(f"\nUsing Groq API with {model_name}...\n")
  client = Groq()

  results = []

  for idx, words in enumerate(test_cases, 1):
    words_str = ", ".join(words)

    print(f"Test {idx}: {words_str}")

    start_time = time.time()
    response = client.chat.completions.create(
      model=model_id,
      messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"So'zlar: {words_str}\nJumla:"}
      ],
      max_tokens=max_tokens,
      temperature=temperature
    )

    elapsed = time.time() - start_time

    sentence = response.choices[0].message.content.strip()

    if "Jumla:" in sentence:
        sentence = sentence.split("Jumla:")[-1].strip()
    if "So'zlar:" in sentence:
        sentence = sentence.split("So'zlar:")[0].strip()

    print(f"Output: {sentence}")
    print(f"Time: {elapsed:.2f}s\n")

    results.append({
      "model": model_name,
      "test_id": idx,
      "input_words": words_str,
      "output_sentence": sentence,
      "time_seconds": round(elapsed, 2)
    })

  return results
  
# https://huggingface.co/uzlm/alloma-3B-Instruct
# https://huggingface.co/uzlm/alloma-1B-Instruct
def test_alloma_model(model_name, model_id, test_cases, max_tokens=50):
   """Test Allome model using transfomers"""
   print(f"\nLoading {model_name}...")

  #  if torch.backends.mps.is_available():
  #     device = torch.device("mps")
  #     dtype = torch.float16  # MPS doesn't support bfloat16 yet
   if torch.cuda.is_available():
      device = torch.device("cuda")
      dtype = torch.bfloat16
   else:
      device = torch.device("cpu")
      dtype = torch.float32

   print(f"Using device: {device}")

   PATTERN  = r"[’‘‚‛ʻʼʽʾʿˈˊˋˌˍ'\']"

   tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
   tokenizer.padding_side = "left"

   if device == "mps":
      pass

   model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)
   
   print(f"{model_name} loaded!\n")

   EOT = "<|eot_id|>"
   SYSTEM = (
        f"{tokenizer.bos_token}<|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant<|eot_id|>"
    )
   
   def create_prompt(user_text):
        return (
            SYSTEM +
            "<|start_header_id|>user<|end_header_id|>\n" +
            f"{user_text}{EOT}" +
            "<|start_header_id|>assistant<|end_header_id|>"
        )
   
   results = []

   for idx, words in enumerate(test_cases, 1):
        words_str = ", ".join(words)
        
        user_input = f"{SYSTEM_PROMPT}\n\nSo'zlar: {words_str}\nJumla:"
        
        # replace apostrophes for better tokenization
        clean_text = re.sub(PATTERN, "APST", user_input)
        
        print(f"Test {idx}: {words_str}")
        
        start_time = time.time()
        
        # encode and generate
        enc = tokenizer(create_prompt(clean_text), return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_tokens,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids(EOT),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        
        elapsed = time.time() - start_time
        
        # decode output
        txt = tokenizer.decode(out[0], skip_special_tokens=False)
        txt = txt.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
        sentence = txt.split(EOT, 1)[0].replace("APST", "'").strip()
        
        # clean response
        if "Jumla:" in sentence:
            sentence = sentence.split("Jumla:")[-1].strip()
        if "So'zlar:" in sentence:
            sentence = sentence.split("So'zlar:")[0].strip()
        
        print(f"Output: {sentence}")
        print(f"Time: {elapsed:.2f}s\n")
        
        results.append({
            'model': model_name,
            'test_id': idx,
            'input_words': words_str,
            'output_sentence': sentence,
            'time_seconds': round(elapsed, 2)
        })
    
   return results
  

# tests ollama models
def test_ollama_model(model_name, model_id, test_cases, temperature=0.7, max_tokens=50):
   """Test Ollama model"""
   print(f"\nUsing Ollama with {model_name}...\n")

   results = []

   for idx, words in enumerate(test_cases, 1):
      words_str = ", ".join(words)

      print(f"Test {idx}: {words_str}")

      start_time = time.time()

      try:
         response = ollama.chat(
            model=model_id,
            messages=[
               {
                  "role": "system",
                  "content": SYSTEM_PROMPT
               },
               {
                  "role": "user",
                  "content": f"So'zlar: {words_str}\nJumla:"
               }
            ],
            options={
               "temperature": temperature,
               "num_predict": max_tokens
            }
         )

         elapsed = time.time() - start_time

         sentence = response["message"]["content"].strip()

         if "Jumla:" in sentence:
            sentence = sentence.split("Jumla:")[-1].strip()
         if "So'zlar:" in sentence:
            sentence = sentence.split("So'zlar:")[0].strip()
            
         print(f"Output: {sentence}")
         print(f"Time: {elapsed:.2f}s\n")

         results.append({
            "model": model_name,
            "test_id": idx,
            "input_words": words_str,
            "output_sentence": sentence,
            "time_seconds": round(elapsed, 2)
         })
      except Exception as e:
         print(f"Error: {e}\n") 
         results.append({
              'model': model_name,
              'test_id': idx,
              'input_words': words_str,
              'output_sentence': f"ERROR: {str(e)}",
              'time_seconds': 0.0
          })
         
   return results
        

def ensure_ollama_running():
   """Check if Ollama is running, if not provide instructions"""
   try:
      result = subprocess.run(['ollama', 'list'],
                              capture_output=True,
                              text=True,
                              timeout=5
                              )
      if result.returncode == 0:
         print("Ollama service is running")
         return True
      else:
         print("Ollama is installed but not responding")
         return False
   except FileNotFoundError:
      print("ERROR: Ollama is not installed")
      print("Install from: https://ollama.com")
      return False
   except subprocess.TimeoutExpired:
      print("ERROR: Ollama service is not responding")
      print("Start it with: ollama serve")
      return False
   except Exception as e:
      print(f"ERROR checking Ollama {e}")
      return False


# save results
def save_to_csv(all_results, filename=None):
    """Save results in pivot format for easy model comparison"""
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'llm_comparison_{timestamp}.csv'
    
    # Group results by test_id
    grouped = {}
    models = set()
    
    for result in all_results:
        test_id = result['test_id']
        input_words = result['input_words']
        model = result['model']
        output = result['output_sentence']
        time_sec = result['time_seconds']
        
        models.add(model)
        
        if test_id not in grouped:
            grouped[test_id] = {
                'input_words': input_words,
                'models': {}
            }
        
        # Combine output and time
        grouped[test_id]['models'][model] = f"{output} ({time_sec}s)"
    
    # Sort models for consistent column order
    sorted_models = sorted(models)
    
    # Write to CSV
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['test_id', 'input_words'] + sorted_models
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for test_id in sorted(grouped.keys()):
            row = {
                'test_id': test_id,
                'input_words': grouped[test_id]['input_words']
            }
            
            # Add each model's output
            for model in sorted_models:
                row[model] = grouped[test_id]['models'].get(model, 'N/A')
            
            writer.writerow(row)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {filename}")
    print(f"{'='*80}\n")
    
    return filename

  

def main():
  all_results = []

  if MLX_AVAILABLE:
    try:
          mlx_results = text_mlx_model(
          model_name="Qwen2.5-1.5B-4bit",
          model_id="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
          test_cases=TEST_CASES,
          temperature=0.1,
          max_tokens=21
          )
          all_results.extend(mlx_results)
    except Exception as e:
        print(f"Error testing MLX model: {e}")
    
# lagged my pc
  # if TRANSFORMERS_AVAILABLE:
  #     try:
  #         alloma_results = test_alloma_model(
  #             model_name="Alloma-3B-Instruct",
  #             model_id="uzlm/alloma-3B-Instruct",
  #             test_cases=TEST_CASES,
  #             max_tokens=50
  #         )
  #         all_results.extend(alloma_results)
  #     except Exception as e:
  #         print(f"Error testing Alloma 3B: {e}\n")

  if TRANSFORMERS_AVAILABLE:
      try:
          alloma_results = test_alloma_model(
              model_name="Alloma-1B-Instruct",
              model_id="uzlm/alloma-1B-Instruct",
              test_cases=TEST_CASES,
              max_tokens=21
          )
          all_results.extend(alloma_results)
      except Exception as e:
          print(f"Error testing Alloma 1B: {e}\n")


  if GROQ_AVAILABLE:
     try:
          groq_results = test_groq_model(
          model_name="Llama-3.3-70B",
          model_id="llama-3.3-70b-versatile",
          test_cases=TEST_CASES,
          temperature=0.1,
          max_tokens=21
          )
          all_results.extend(groq_results)
     except Exception as e:
        print(f"Error testing Groq model: {e}")

  if OLLAMA_AVAILABLE:
     if not ensure_ollama_running():
        print("Skipping Ollama tests - service not available")
     else:
        ollama_models = [
        {"name": "Qwen2.5-1.5B-Ollama", "id": "qwen2.5:1.5b"},
        {"name": "Qwen2.5-3B-Ollama", "id": "qwen2.5:3b"}
        # {"name": "Llama3.2-3B-Ollama", "id": "llama3.2:3b"},
        # {"name": "Gemma2-2B-Ollama", "id": "gemma2:2b"}
     ]

     for model_config in ollama_models:
        try:
           ollama_results = test_ollama_model(
              model_name=model_config["name"],
              model_id=model_config["id"],
              test_cases=TEST_CASES,
              temperature=0.1,
              max_tokens=21
           )
           all_results.extend(ollama_results)
        except Exception as e:
           print(f"Error testing {model_config['name']}: {e}\n")


  if all_results:
     save_to_csv(all_results)
     print("Testing complete! Check the CSV file for comparison.")
  else:
     print("No results to save.")
      

if __name__ == "__main__":
  main()


