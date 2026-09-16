from google import genai
import os


# Get API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please set your API key in PowerShell first."
    )


# Create Gemini client
client = genai.Client(api_key=api_key)


# Send a simple request
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Explain cardiovascular disease in two simple sentences."
)


print("\nGemini response:")
print(response.text)

