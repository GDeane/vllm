# Disaggregated Prefill V1 (Hybrid Prefill Mode)

This example mirrors `disaggregated-prefill-v1`, but the prefill phase runs the new hybrid-prefill optimizations (`prefill_mode=True`), which keeps each request as a single chunk and enables FFN chunking automatically.

## Files

- `run.sh` - A helper script that will run `prefill_example.py` and `decode_example.py` sequentially.
    - Make sure you are in the `examples/offline_inference/disaggregated-prefill-v1` directory before running `run.sh`.
- `prefill_example.py` - Hybrid-prefill version of the prefill script. It saves the KV state to the `local_storage` directory and the prompts to `output.txt`.
- `decode_example.py` - A script which performs decode only, loading the KV state from the `local_storage` directory and the prompts from `output.txt`.
