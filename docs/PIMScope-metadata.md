# PIMScope capability and power metadata

PIMScope emits machine-readable capability and power-accounting fields. Consumers
should branch on those fields rather than parsing explanatory text.

## Backend capabilities

- `status`: `supported` or `experimental`.
- `paper_artifact_backend`: whether the backend is used for fixed paper artifacts.
- `validated_topology` and `validated_rank_counts`: tested public scope.
- `host_access_timing_vocabulary`: standard-specific host command vocabulary.
- `paper_artifact_backend_name`: backend to use for paper reproduction.
- `subchannel_model`: modeled versus physical LPDDR6 sub-channel behavior.
- `standard_power_calibration`: calibration level of standard-memory energy.
- `pim_energy_method`: source of incremental PIM event energy.

LPDDR6PIM is experimental. One modeled channel represents one 12-bit
sub-channel; independent paired-sub-channel scheduling is not modeled. Its
standard energy uses the DRAMPower v6.2.0 LPDDR6 test fixture rather than a
production-device datasheet. Paper reproduction remains pinned to LPDDR5PIM.

## Power accounting

- `status` and `model` identify the accounting contract.
- `equation` and `energy_scope` identify included terms.
- `standard_power_source` records source, version, and calibration.
- `pim_energy_method` identifies incremental PIM event accounting.
- `standard_power_calibrated_to_device` is the device-calibration boundary.
- `metadata_documentation` links back to this contract.

LPDDR5PIM uses the paper two-layer decomposition: configured standard-memory
energy plus PIM command-event terms. LPDDR6PIM uses the experimental DRAMPower
test profile plus the same event-coefficient method.
