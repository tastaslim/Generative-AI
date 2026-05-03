# Document Loader

- Document loaders provide a standard interface for reading data from different sources (such as Slack, Notion, or
  Google Drive). This ensures that data(of any type like PDF, CSV, database, website etc.) can be handled consistently
  regardless of the source by converting them into one consistent object.
- All document loaders implement the **BaseLoader** interface. Each document loader may define its own parameters, but
  they share a common API:
    1. **load()** – Loads all documents at once.
    2. **lazy_load()** – Streams documents lazily, useful for large datasets.

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader("")  # Integration-specific parameters here
# Load all documents
documents = loader.load()
# For large datasets, lazily load documents
for document in loader.lazy_load():
	print(document)
```

- Read [this](https://docs.langchain.com/oss/python/integrations/document_loaders) for more details.