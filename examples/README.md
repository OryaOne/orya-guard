# Example files

These files are designed for first-time `orya-guard` users.

They are intentionally small so you can read them quickly, run the CLI immediately, and understand what each command is demonstrating.

## Files

- `dataset_with_issues.csv`
  Demonstrates `check-dataset`.
  Includes one duplicate row, one nullable column with a `50%` null ratio, and two constant columns.

- `train.csv`
  Reference dataset for `compare`.
  Represents a simple training snapshot with stable categorical values and numeric ranges.

- `candidate.csv`
  Candidate dataset for `compare`.
  Demonstrates null ratio changes, unseen categorical values (`volos`), and numeric drift signals relative to `train.csv`.

- `inference_schema.json`
  Simple schema used by `check-inference-payload`.
  Declares required fields and simple field types.

- `inference_payload_valid.json`
  Valid example payload.
  Demonstrates a passing inference payload check.

- `inference_payload_invalid.json`
  Invalid example payload.
  Demonstrates missing required fields, unexpected fields, and type mismatches.

## Recommended demo flow

```bash
orya-guard check-dataset examples/dataset_with_issues.csv
orya-guard compare examples/train.csv examples/candidate.csv
orya-guard check-inference-payload examples/inference_payload_valid.json --schema examples/inference_schema.json
```
