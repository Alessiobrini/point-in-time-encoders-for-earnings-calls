"""Which tokenizer and model classes the encoder loader uses for a registry entry."""
import pytest
import transformers

from ecalls.chunk import registry as R
from ecalls.embed import encode as E


def _entry(extra=None):
    return R.Entry("m", R.ATEMPORAL, "huggingface:x/y", 768, None, extra=extra or {})


def test_the_auto_classes_by_default():
    assert E.auto_classes(_entry()) == (transformers.AutoTokenizer, transformers.AutoModel)


def test_a_named_architecture_gets_its_own_classes():
    tok, model = E.auto_classes(_entry({"hf_classes": "bert"}))
    assert tok is transformers.BertTokenizerFast and model is transformers.BertModel


def test_an_unknown_name_is_refused_rather_than_imported():
    with pytest.raises(ValueError, match="not one of"):
        E.auto_classes(_entry({"hf_classes": "os.system"}))


def test_finbert_is_registered_with_bert_classes_and_lowercasing():
    e = R.load().by_name("finbert-pretrain")
    assert e.kind == R.ATEMPORAL
    assert e.extra["hf_classes"] == "bert" and e.extra["tokenizer_kwargs"] == {"do_lower_case": True}
