from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.responses.create(
    model="gpt-5.1", input="Write a two paragraph story about John and his robots.")
print(response.output_text)
