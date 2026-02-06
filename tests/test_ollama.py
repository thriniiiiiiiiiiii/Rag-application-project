"""
Test Ollama Local LLM (100% FREE!)
"""
import requests
import json

print("="*70)
print("🧪 TESTING OLLAMA (FREE LOCAL LLM)")
print("="*70)

# Check if Ollama is running
try:
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        models = response.json().get('models', [])
        print(f"\n✅ Ollama is running!")
        print(f"📦 Available models: {[m['name'] for m in models]}")
    else:
        print("\n❌ Ollama not running")
        print("Start Ollama and run: ollama pull llama3.2")
        exit(1)
except:
    print("\n❌ Cannot connect to Ollama")
    print("1. Download from: https://ollama.com/download")
    print("2. Install Ollama")
    print("3. Run: ollama pull llama3.2")
    exit(1)

# Test generation
print("\n🔄 Testing text generation...")

data = {
    "model": "llama3.2",
    "prompt": "Say 'Hello! Ollama works!' and nothing else.",
    "stream": False
}

try:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=data
    )
    
    result = response.json()
    answer = result['response']
    
    print("\n" + "="*70)
    print("✅ SUCCESS! OLLAMA IS WORKING!")
    print("="*70)
    print(f"\n🤖 Response: {answer}")
    print(f"\n💰 Cost: $0.00 (FREE!)")
    print("\n🎉 You can build RAG for FREE with Ollama!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")