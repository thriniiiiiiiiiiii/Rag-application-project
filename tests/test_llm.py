"""
Test OpenAI API Connection
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Check if API key is properly loaded"""
    print("="*60)
    print("🔑 CHECKING API KEY")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found!")
        print("\n📝 To fix:")
        print("1. Open .env file")
        print("2. Add this line (with your actual key):")
        print("   OPENAI_API_KEY=sk-proj-your-key-here")
        print("3. Save the file")
        return False
    
    if not api_key.startswith("sk-"):
        print("❌ ERROR: Invalid API key format")
        print(f"   Your key starts with: {api_key[:5]}")
        print("   It should start with: sk-proj- or sk-")
        return False
    
    print("✅ API key found!")
    print(f"   Key preview: {api_key[:20]}...{api_key[-4:]}")
    return True

def test_openai_connection():
    """Test actual API call"""
    print("\n" + "="*60)
    print("🌐 TESTING OPENAI CONNECTION")
    print("="*60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created")
    except Exception as e:
        print(f"❌ ERROR creating client: {e}")
        return False
    
    print("\n🔄 Sending test request...")
    print("   (This will use ~50 tokens = $0.0001)")
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API test successful!' and nothing else."}
            ],
            temperature=0,
            max_tokens=20
        )
        
        answer = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost = tokens * 0.000002
        
        print("\n" + "="*60)
        print("✅ SUCCESS! API IS WORKING!")
        print("="*60)
        print(f"\n🤖 LLM Response: {answer}")
        print(f"📊 Tokens used: {tokens}")
        print(f"💰 Cost: ${cost:.6f}")
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ API CALL FAILED")
        print("="*60)
        print(f"\nError: {str(e)}")
        
        error_msg = str(e).lower()
        
        if "authentication" in error_msg or "api key" in error_msg:
            print("\n💡 AUTHENTICATION ERROR")
            print("   - Your API key may be invalid or expired")
            print("   - Check: https://platform.openai.com/api-keys")
            print("   - Create a new key if needed")
            
        elif "quota" in error_msg or "insufficient" in error_msg:
            print("\n💡 QUOTA ERROR")
            print("   - You need to add credits to your account")
            print("   - Go to: https://platform.openai.com/account/billing")
            print("   - Add at least $5 in credits")
            
        elif "rate limit" in error_msg:
            print("\n💡 RATE LIMIT ERROR")
            print("   - Wait 20 seconds and try again")
            
        else:
            print("\n💡 UNKNOWN ERROR")
            print("   - Check your internet connection")
            print("   - Try again in a few seconds")
        
        return False

def main():
    print("\n" + "🚀"*30)
    print("OPENAI API TEST SCRIPT")
    print("🚀"*30 + "\n")
    
    # Test 1: Check API key
    if not test_api_key():
        print("\n❌ Fix API key issue and try again")
        return
    
    # Test 2: Test connection
    if not test_openai_connection():
        print("\n❌ Fix connection issue and try again")
        return
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
    print("\n✅ You're ready to build your RAG application!")

if __name__ == "__main__":
    main()