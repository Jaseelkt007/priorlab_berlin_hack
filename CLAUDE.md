# priorlab_berlin_hack

Our project for the **Prior Labs track, "TabPFN for spatial awareness"**, at HackLab #2 (Berlin Robotics & AI Hackathon, CIC Berlin, 21-22 Nov 2026). The goal: use TabPFN's in-context learning to help robots and agents predict where things are, where they're going and what happens next.

- **Hard deadline: Sun 22 Nov 2026, 16:30 CET**, via https://tally.so/r/GexoxQ. Every submission needs a 3-minute video (pitch, how it works, demo; a person on camera), a public GitHub repo with docs, and slides for a 5-minute pitch.
- **This repo is public at submission:** no secrets, tokens or coupon codes in git. `TABPFN_TOKEN` lives in `.env`, which is ignored.
- Full brief, schedule and setup: `docs/HACKATHON.md`.

<!-- potato-start:begin mode=hackathon version=1 -->
## How we build here (hackathon mode)

Speed first, but a demo that breaks on stage is the expensive failure.

- **Prove the demo path before saying done.** Run the real thing once (the page, the CLI, the endpoint) and check what a judge will see. "It compiles" is not done.
- **One way to do each thing.** Don't leave a second copy or a dead attempt in the code; delete it. Agents copy whatever they find.
- **Quick fixes are allowed, and each one goes in `DEBT.md`** (one line: file, what was hacked, the real fix). A fixed sleep, a hard-coded value, a skipped error path all count.
- Comments are fine. No pre-commit checks, no worktrees.
- If this project lives past the event, run `/potato-start product` to tighten up; it starts from `DEBT.md`.
<!-- potato-start:end -->
