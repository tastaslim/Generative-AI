from tiktoken import Encoding, encoding_for_model
inputText: str = "Who is the prime minister of India?"
tokenizer: Encoding = encoding_for_model(model_name="gpt-5")
tokens: list[int] = tokenizer.encode(inputText)
print(tokens)

originalText: str = tokenizer.decode(tokens)
print(originalText)