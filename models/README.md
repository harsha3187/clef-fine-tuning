# models/

Fine-tuned model weights and their model card. The weights themselves are not stored in git, and everything under `models/` except the model card, checksums and configuration is ignored.

| Folder | What | Size |
|---|---|---|
| [clef-flash-capability-ft-v3](clef-flash-capability-ft-v3/MODEL_CARD.md) | LoRA adapter and head that predict how much model a request needs (low, medium, high) | 800 MB |

The weights are about 800 MB, so they are not in the repository. Download `adapter_model.safetensors` and `joint_head.safetensors` from the
repository's Releases page when they are attached there, or train your own router (see the main README), and place them like this:

```text
models/clef-flash-capability-ft-v3/adapter/adapter_model.safetensors
models/clef-flash-capability-ft-v3/joint_head.safetensors
```

Then check every file against `SHA256SUMS`:

```sh
python -m routing_lab.verify_model models/clef-flash-capability-ft-v3
```

The containers run the same check at startup. Nothing else in the repository needs the weights: the app calls Clef on an Azure endpoint.

The base model (`Cloudflare/clef-flash`, 18 GB) is not stored here. It is public and is downloaded on first use at the revision this code pins.
