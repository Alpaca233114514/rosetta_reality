# GPU audit 006: admission caught renewed allocation

The original instance was reopened after run 005's protected shutdown.
Its current RTX 4090 D allocation is GPU-6cca7cea-b646-2955-b4ba-cbe17df292c4,
idle at 1 MiB. The prior template still declared GPU-8a632f30-... and the
supervisor correctly rejected it before creating run 006 or loading weights.
The failed admission log is retained in workspace 20260914T082615Z-95cf9cf9483b-187b723865bc.

Template 010 records the current allocation and otherwise preserves amendment
007's exact two-update, numerical, data, Gate and resource bounds. The code and
strict UUID comparison are unchanged. Same original instance, no new rental.
