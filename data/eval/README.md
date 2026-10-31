# Evaluation set

`gold.jsonl` is created by `fcopilot gen-gold` and reviewed with `fcopilot review`. It is the only data file committed to the repository, so results can be reproduced and audited.

Each line is a `GoldQuestion`:

| field | meaning |
|---|---|
| `qid` | stable id, prefixed by type (`si`, `ta`, `mu`, `un`) |
| `question` | the analyst question |
| `answer` | reference answer |
| `qtype` | `single_hop`, `table`, `multi_hop` or `unanswerable` |
| `evidence` | list of `{doc_id, start, end, quote}`: character spans in the parsed filing |
| `verified` | `true` once a human accepted or edited the item |

Evidence spans index into `data/parsed/<doc_id>.json["text"]`, which `make data` regenerates deterministically from the EDGAR HTML.
