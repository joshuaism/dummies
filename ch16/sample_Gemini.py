from google import genai
import os

# create a client
# get api key from https://aistudio.google.com
gemini_api_key = os.environ.get('GEMINI_KEY')
client = genai.Client(api_key=gemini_api_key)

# Fetch metadata about available models
models = client.models.list()

for m in models:
    print("Model: ", m.name)
    print(" Capabilites: ", m.supported_actions)

# Single prompt to Gemini
response = client.models.generate_content(
    model="models/gemini-3.5-flash-lite",
    contents="Write a two paragraph story about John and his robots."
)

# Print output
print(response.text)
