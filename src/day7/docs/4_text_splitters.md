# Text Splitters

Text splitters break large docs into smaller chunks that will be retrievable individually and fit within model context
window limit. There are several strategies for splitting documents, each with its own advantages.

For most use cases, start with the **RecursiveCharacterTextSplitter**. It provides a solid balance between keeping
context intact and managing chunk size. This default strategy works well out of the box, and you should only consider
adjusting it if you need to fine-tune performance for your specific application.

**Read [this](https://docs.langchain.com/oss/python/integrations/splitters) doc for more details**