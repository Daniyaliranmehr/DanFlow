## extract_zip()

Extracts a ZIP archive to a specified directory.

### Parameters

#### `zip_path` : `str | Path`
Path to the ZIP archive.

#### `output_path` : `str | Path`
Directory where the archive will be extracted.

### Returns

`None`

### Example

```python
from danflow.data.io import extract_zip

extract_zip(
    "dataset.zip",
    "data/"
)