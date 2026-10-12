# models/

Fine-tuned model weights that ship with this repository. Everything else under `models/` is ignored by git.

| Folder | What | Size |
|---|---|---|
| [clef-flash-capability-ft-v3](clef-flash-capability-ft-v3/MODEL_CARD.md) | LoRA adapter and head that predict how much model a request needs (low, medium, high) | 800 MB |

The weights are stored with **Git LFS**. After cloning:

```sh
git lfs install      # once per machine
git lfs pull         # fetch the real files instead of the small pointer files
python -m routing_lab.verify_model models/clef-flash-capability-ft-v3
```

The last command checks every file against `SHA256SUMS` and says so if a file is still an LFS pointer. The containers run the same check at
startup.

The base model (`Cloudflare/clef-flash`, 18 GB) is not stored here. It is public and is downloaded on first use at the revision this code pins.
