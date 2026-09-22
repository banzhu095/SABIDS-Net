# Checkpoint binding

`tools/bind_training_checkpoint_evidence.py` derives a new
`checkpoint_binding_best.json`; it never edits the initial inventory, history,
run metadata, or checkpoint. Formal best-D1 preflight now requires both the
training-start inventory and this derived binding. Any source SHA, epoch,
monitor, run, protocol, split, manifest, pixel inventory, or checkpoint
identity mismatch blocks use.

The binding is a local hash chain, not an external signature. Run it on the
cloud assets and retain the original files. Until that command passes, best D1
and every D2 config initialized from it remain blocked.

