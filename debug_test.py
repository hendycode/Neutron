from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Check API key
api_key = os.getenv("ANTHROPIC_API_KEY")
print(f"API Key found: {api_key is not None}")
if api_key:
    print(f"API Key starts with: {api_key[:10]}...")
else:
    print("❌ No API key found!")
    exit(1)

# Test 1: Basic initialization
print("\n=== Test 1: Basic LLM Initialization ===")
try:
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        temperature=0
    )
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ LLM initialization failed: {e}")
    exit(1)

# Test 2: Simple invoke with string
print("\n=== Test 2: Simple String Invoke ===")
try:
    response = llm.invoke("Hello, please respond with just 'Hi there!'")
    print("✅ Simple invoke successful")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"❌ Simple invoke failed: {e}")
    print(f"Error type: {type(e)}")

# Test 3: Invoke with HumanMessage
print("\n=== Test 3: HumanMessage Invoke ===")
try:
    message = HumanMessage(content="Hello, please respond with just 'Hi there!'")
    response = llm.invoke([message])
    print("✅ HumanMessage invoke successful")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"❌ HumanMessage invoke failed: {e}")
    print(f"Error type: {type(e)}")

# Test 4: Check model name
print("\n=== Test 4: Try Different Model ===")
try:
    llm_alt = ChatAnthropic(
        model="claude-3-haiku-20240307",  # Different model
        api_key=api_key,
        temperature=0
    )
    response = llm_alt.invoke("Hello, please respond with just 'Hi there!'")
    print("✅ Alternative model successful")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"❌ Alternative model failed: {e}")

print("\n=== Environment Info ===")
print(f"Python version: {os.sys.version}")
try:
    import langchain_anthropic
    print(f"langchain_anthropic version: {langchain_anthropic.__version__}")
except:
    print("Could not get langchain_anthropic version")

try:
    import anthropic
    print(f"anthropic version: {anthropic.__version__}")
except:
    print("Could not get anthropic version")