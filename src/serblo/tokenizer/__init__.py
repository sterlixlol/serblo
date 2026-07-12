"""M4: 32k byte-level BCS BPE, trained on Latin-normalized corpus. See PRD 7.2."""


def load_tokenizer(path="data/tokenizer/tokenizer.json"):
    """Load the saved tokenizer.json artifact. Lazy import so importing the package stays cheap."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))
