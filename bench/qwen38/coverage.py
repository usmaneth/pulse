"""Coverage of a draft vocabulary on held-out agent output."""
import json, sys, collections
from tokenizers import Tokenizer
TOK = "/mnt/models/hf/hub/models--Mia-AiLab--Qwen3.8-Flash-Next-NVFP4/snapshots/925d7be6c14c6c9442ef83e8f05b5a3c39304f69/tokenizer.json"
tok = Tokenizer.from_file(TOK)
c = collections.Counter(); n = 0
for line in open("heldout.jsonl"):
    ids = tok.encode(json.loads(line)["text"], add_special_tokens=False).ids
    c.update(ids); n += len(ids)
for path in sys.argv[1:]:
    v = set(int(x) for x in open(path) if x.strip())
    cov = sum(x for k, x in c.items() if k in v) / n
    miss = collections.Counter({k: x for k, x in c.items() if k not in v})
    print(f"{path.split('/')[-1]}: size {len(v):,}  coverage {cov:.4%}  missed-occurrences {n - int(cov*n):,}/{n:,}")
    print("   top missed:", [(tok.id_to_token(k), x) for k, x in miss.most_common(15)])
