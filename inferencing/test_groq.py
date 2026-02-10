"""
required package: pip install groq

ON APPLE SILLICON
CONDA ENV

1. nano ~/.zshrc
inside of of .zshrc add this statement below
2. export GROQ_API_KEY="your_api_key_here"
3. source ~/.zshrc
4. pip install groq
5. python -c "import groq; print('Groq installed')"
6. conda env config vars set GROQ_API_KEY="your_api_key_here"
7. conda deactivate
8. conda activae llm-uzslr-signs
9. echo $GROQ_API_KEY

"""

from groq import Groq

client = Groq()

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Explain why fast inference matters."}
    ],
)

print(response.choices[0].message.content)
