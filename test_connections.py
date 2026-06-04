from dotenv import load_dotenv
import os
from openai import OpenAI
from groq import Groq

load_dotenv()

# Test OpenAI embeddings
print("Testing OpenAI...")
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding = openai_client.embeddings.create(
    input="ibuprofen side effects",
    model="text-embedding-3-small"
)
print(f"✅ OpenAI OK — embedding size: {len(embedding.data[0].embedding)}")

# Test Groq LLM
print("Testing Groq...")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say hello in one word"}]
)
print(f"✅ Groq OK — response: {response.choices[0].message.content}")

print("\n✅ All connections working!")