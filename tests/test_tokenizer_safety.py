from pocketsql.data.validate import is_read_only_select
from pocketsql.model.tokenizer import ByteTokenizer
from pocketsql.training.dataset import encode_record


def test_byte_tokenizer_round_trips_utf8_and_special_tokens():
    tokenizer = ByteTokenizer()
    text = "<bos><schema>caf" + chr(0xE9) + "</schema><sql>SELECT " + chr(0x3BB) + ";</sql><eos>"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_loss_mask_excludes_prompt_tokens():
    tokenizer = ByteTokenizer()
    record = {"schema_sql": "CREATE TABLE x (id INTEGER);", "question": "show id", "sql": "SELECT id FROM x;"}
    ids, mask = encode_record(record, tokenizer, 256)
    sql_start = ids.index(tokenizer.token_to_id["<sql>"])
    assert not any(mask[:sql_start])
    assert all(mask[sql_start:])


def test_unsafe_and_multi_statement_sql_is_rejected():
    assert is_read_only_select("SELECT * FROM customers;")
    assert not is_read_only_select("SELECT * FROM customers; DROP TABLE customers;")
    assert not is_read_only_select("DELETE FROM customers")