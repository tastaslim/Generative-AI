from typing import List
from tiktoken import encoding_for_model, Encoding
inputText: str = "Who is the Captain of RCB?"
tokenizer: Encoding = encoding_for_model(model_name="gpt-5")
tokens: List[int] = tokenizer.encode(inputText)
print(tokens)
originalText: str = tokenizer.decode(tokens)
print(originalText)
