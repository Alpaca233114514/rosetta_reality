# Historical SmolVLA source bytes

These ten text files are data fixtures, never imported or executed. Each filename
is the original plan's SHA-256. `manifest.json` records the local Git blob or
previously retained source snapshot and its exact byte count. `.gitattributes`
prevents newline conversion on checkout.

`tests/historical_smolvla_sources.py` constructs a temporary protocol filesystem,
checks every byte against the original plan, then points only the tested protocol
module's file location there. Production modules and old plans are unchanged.
Separate negative tests continue to assert that modified historical bytes and
the current incompatible source tree are rejected.

Zen and vfunfreeze have complete verified implementation fixtures. Two historical
vcdropout files are not recoverable from the inspected local Git history:

- `src/rosetta_reality/vla/training/features.py`: `e0ded450340be7f6808d8f764122dd52e8adf3a964b07f00f0f3ea3dd5fdbe53`
- `src/rosetta_reality/vla/training/plan.py`: `efc70554855d6cfb63c78d180f485d82a94ee7ca3bb053d4e29136e29ddd09d7`

The vcdropout gate unit tests use an explicitly synthetic temporary plan, bound
to actual current file hashes by the existing test factory. They do not certify
historical run replay. The real historical plan must still reject this tree.
No hash-return stub, skipped test or disabled identity check is used.
