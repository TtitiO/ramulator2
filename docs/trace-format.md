# Concrete PIM trace format

PIMScope concrete traces are JSON Lines files. The first line is a header; each
following line is one opcode record. The header must identify `dram_class` and
`schema_version`.

Supported schemas:

- `LPDDR5PIM`: `lpddr5-pim-opcode-v0.2`
- `LPDDR6PIM`: `lpddr6-pim-opcode-v0.1`

Validate a trace against a resolved manifest without running replay:

```bash
pimscope validate-trace TRACE.jsonl --config MANIFEST.json
```

## Record classes

| Record | Backend meaning |
| --- | --- |
| `READ`, `WRITE` | Host request, lowered to the backend read/write request type. |
| `PIM_MAC` | Per-bank PIM request, lowered to `PIMCompute`. |
| `PIM_BCAST` | Direct rank/broadcast command, lowered to `PIMLoadAll`. |
| `PIM_MAC_AB` | Rank/all-bank PIM request, lowered to `PIMComputeAll`. |
| `SB`, `HAB`, `HAB_PIM` | Direct DRAM mode commands. |

`repeat` expands one record into repeated records. Validators enforce positive
repeat values, address bounds, opcode/backend compatibility, mode ordering, and
configured expanded-record limits. LPDDR6PIM uses its own command vocabulary and
schema; LPDDR5PIM opcodes are rejected for LPDDR6PIM traces.

Address fields are byte addresses unless the record uses an explicit
`addr_vec`. Byte addresses are decomposed using the resolved organization,
transaction size, and mapping version in the manifest. All-bank records use the
resolved rank scope; they do not imply cross-rank scheduling.

The trace validator reports record count, expanded count, schema, backend,
resolved capacity, and mapping version. It does not claim numerical model
correctness, application-runtime replay, or silicon-faithful timing.
