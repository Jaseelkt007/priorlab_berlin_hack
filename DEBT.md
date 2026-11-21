# Debt

Quick fixes taken to move fast. One line each: file, what was hacked, what the real fix is.
`/potato-start product` starts from this list.

- `collect/preview_flight.py`: re-implements acme's SinglePrecision/CanonicalSpec wrappers and flybody's TestPolicyWrapper inline, because `dm-acme` needs `launchpad`. Real fix: a pinned env file (environment.yml) that installs flybody[tf] reproducibly, then use their wrappers.
- `collect/closed_loop.py`: context capped at 256 steps (3,072 rows) so TabPFN 3.5 Fast fits the GTX 1050's 3 GB; 512 steps runs out of memory. Real fix: run on the RTX 4050 (6 GB) with 1-2k steps, or kv_cache_precision / memory_saving_mode.
