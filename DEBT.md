# Debt

Quick fixes taken to move fast. One line each: file, what was hacked, what the real fix is.
`/potato-start product` starts from this list.

- `collect/preview_flight.py`: re-implements acme's SinglePrecision/CanonicalSpec wrappers and flybody's TestPolicyWrapper inline, because `dm-acme` needs `launchpad`. Real fix: a pinned env file (environment.yml) that installs flybody[tf] reproducibly, then use their wrappers.
