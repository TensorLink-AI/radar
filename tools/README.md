# RADAR verification tools

Standalone scripts you can run to prove that a RADAR experiment was honestly
recorded. They depend only on `shared.substrate` (signing/verification) and
`shared.hippius_client` (IPFS download). No RADAR-internal code beyond that
— fork them freely.

## `verify_radar_experiment.py`

Fetches an experiment from a RADAR API, then for each substrate CID it
claims to be backed by, downloads the Hippius bundle, sr25519-verifies the
signature, and compares the bundle's signed metric record with what the
API returned. Discrepancies — wrong signature, missing record, mismatched
metric — are surfaced as a per-CID list in the report.

### Usage

```bash
python -m tools.verify_radar_experiment \
    --api-url https://radar.example.com \
    --experiment-id 12345
```

`--hippius-ipfs` overrides the IPFS gateway (default: the public Hippius
gateway). Point it at a local IPFS node or any other IPFS-compatible
gateway you trust:

```bash
python -m tools.verify_radar_experiment \
    --api-url https://radar.example.com \
    --experiment-id 12345 \
    --hippius-ipfs http://localhost:5001
```

### Exit codes

| Code | Meaning                                                            |
|------|--------------------------------------------------------------------|
| 0    | Every substrate record was fetched, signed correctly, and matched the API |
| 1    | Verification failed — signature invalid, missing record, or metric mismatch |
| 2    | Experiment exists but has no `substrate_cids` (nothing to verify) |
| 3    | API unreachable                                                    |

### What it actually checks

For each CID listed in the experiment's `substrate_cids` array:

1. **Fetchable.** The bundle is downloadable from the IPFS gateway.
2. **Parseable.** The bundle deserialises into the schema declared in
   `shared/substrate.py`.
3. **Signed correctly.** The `(round_id, miner_uid)`-matching record's
   sr25519 signature verifies under the validator hotkey claimed in both
   the bundle and the audit-trail entry.
4. **Matches the API.** Every metric in the signed record equals the same
   metric in the API's `results` dict; `miner_uid` and `miner_hotkey`
   agree across both sources.

If any step fails the per-CID block prints `FAIL` with a `discrepancies`
list. A clean run prints `PASS` for every CID and "All substrate records
verified."

### Why this exists

RADAR's centralised database is convenient but it's a single point of
trust. Substrate bundles on Hippius are a parallel, independently
verifiable record: anyone can pull the raw signed payload, check the
sr25519 signature against the validator's public hotkey, and compare it
to what the API claims. If those two records ever disagree, this tool
makes it loud.

Use this to prove any RADAR experiment was honestly recorded.
