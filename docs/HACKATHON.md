# HackLab #2: Berlin Robotics & AI Hackathon (Prior Labs track)

Source: the organisers' Notion page, pasted 2026-11-21. Coupon and activation codes are left out on purpose: this repo will be public.

## Event

- **When:** Sat 21 Nov 2026, 09:00 to Sun 22 Nov 2026, 19:00 CET. Hacking starts Sat 11:00 and continues overnight.
- **Where:** CIC Berlin, Lohmühlenstraße 65, 12435 Berlin.
- **Hosts:** HackLab & BRISA (Berlin Robotics & Intelligent Systems Association).
- **Discord:** https://discord.gg/x86D9Fkcs (live updates, mentors).

| Day | Time | What |
|---|---|---|
| Sat | 11:00 | Hacking starts |
| Sat | 13:00 / 19:00 | Lunch (1st floor) / Dinner (3rd floor) |
| Sun | 09:00 / 13:30 | Breakfast (3rd floor) / Lunch (1st floor) |
| **Sun** | **16:30** | **Submission deadline (hard)** |
| Sun | 18:00 | Live pitches & award ceremony |

## Our track: Prior Labs, "TabPFN for spatial awareness"

> Use TabPFN's in-context learning to help robots and agents predict **where things are, where they're going and what happens next**. How far can a table take a robot?

- **Prize:** €750 cash.
- **Cost:** TabPFN is free; no paid credits needed.
- **Code:** https://github.com/PriorLabs/TabPFN
- **Docs:** https://docs.priorlabs.ai/quickstart
- **Prior Labs deck:** [`HackLab_Berlin_x_TabPFN_Prior_Labs.pdf`](HackLab_Berlin_x_TabPFN_Prior_Labs.pdf).
- **Judging criteria:** not published on the page. Ask Prior Labs on site or in Discord. Until we hear, aim for: a working demo, a clear use of TabPFN's in-context learning (few-shot, no training loop), and a spatial or robotics story that reads quickly.

### Setup (required)

1. Free account at https://platform.priorlabs.ai, then accept the model license under **Licenses**.
2. Create an API key and export it as `TABPFN_TOKEN` (keep it in `.env`, which git ignores).
3. Install one of:
   - `tabpfn-client`: cloud inference, no GPU. Recommended by Prior Labs.
   - `tabpfn`: local inference on an NVIDIA GPU. The first run downloads the weights. Our laptop has an RTX 4050 with 6 GB, which is fine for small tables.

## What every team submits (Prior Labs falls under "Simulation and Software")

Deadline **Sun 22 Nov, 16:30 CET**, via https://tally.so/r/GexoxQ:

1. **Video, 3 minutes** (unlisted YouTube or a Google Drive link that needs no special access), containing:
   - a clear pitch,
   - an explanation of the solution,
   - a demo or screenshots of it working.
   Strongly recommended: a team member speaking to camera, and the real product on screen.
2. **GitHub repo**, public or accessible to the judges, with the code and technical docs.
3. **Slides** for a 5-minute live pitch on the main stage if we win the track.

Missing any of these can mean disqualification.

## Shared credits (any track)

Claim links are on the Notion page or sent by email. Don't post codes anywhere public.

- **Anthropic:** free Claude API credits.
- **Cognition:** $200 Devin credits per participant (sign up with the email you registered with).
- **Linkup:** web search API for agents (linkup.so).
- **NVIDIA Brev:** $150 GPU credits; the code arrives by email.
- **Nebius Token Factory:** $100 (the sign-up asks for a bank card).

## House rules

- Don't take, open or modify partner hardware; never leave a robot running unattended; report any breakage right away.
- All hardware stays at the venue.
