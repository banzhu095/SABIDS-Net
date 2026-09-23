# D2 teacher evidence audit (2026-09-23)

The attached lightweight archive was audited read-only. It does not contain a
checkpoint, so its recorded checkpoint SHA cannot be recomputed locally.

| Evidence field | safe-current historical value | active lock value | Verifiable/equivalent? | Historical source |
| --- | --- | --- | --- | --- |
| protocol ID | missing | `pku37_binary_v3` | no | `resolved_config.yaml`, protocol audit |
| checkpoint SHA | `230e9379...161b1d53` | n/a | recorded only; checkpoint absent locally | SHA report, `run_metadata.json` |
| best epoch | 32 | n/a | yes; first maximum in 44-row rectangular history | `history.csv`, `run_metadata.json` |
| selection metric | validation vessel soft Dice, 0.7155813716 | same required metric | yes | config, history, metadata |
| train positions | 13 label-eligible positions | 30-position denoising/development pool | not directly comparable; teacher cohort must equal locked `train_segment.csv` and be a subset of the 30 | config/runtime vs active lock |
| validation positions | `0006,0012,0040` | same three | yes | config/runtime vs active lock |
| split-contract SHA | missing | `cc2aee3e...716d90e` | no | config vs active lock |
| data-plan SHA | missing | `9b03235d...c6fa98` | no | config vs active lock |
| manifest SHA | `2e881495...7c4f9e` | locked protocol uses a different manifest set | not equivalent | config/runtime |
| label asset SHA | historical run fingerprint `7b5d9625...2c9a57` | protocol inventory `cc39ab67...a4d26f` | different semantics and not bindable | config/runtime vs active lock |
| input resolution | 512x512 | 512x512 | yes | config vs active lock |
| normalization | fixed | fixed | yes | config vs active lock |
| test opened | historical label inventory includes sealed-test groups | lock says 0 | cannot prove sealed-test exclusion | `label_asset_inventory.json`, active lock |

Decision: **RETRAIN FORMAL TEACHER REQUIRED**. The historical safe-current
checkpoint cannot receive a derived binding because the locked hashes were
never recorded, its 13-position cohort cannot be proven identical to the
current locked `train_segment.csv` from the supplied archive, and its
historical label inventory inspected sealed-test label paths. Adding a current
`protocol_id` would not repair those missing training-time facts.
