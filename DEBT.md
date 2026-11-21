# Debt

Quick fixes taken to move fast. One line each: file, what was hacked, what the real fix is.
`/potato-start product` starts from this list.

- `drone/intruder.py` Radar: noise grows with distance from the ground point (0, 0, 0), but the dish is drawn on the roof at about 14 m (`sim.RADAR_TOP`). Real fix: measure from `RADAR_TOP` and rebuild the data.
- `drone/intruder.py` Flight(guard=True): the X2 defender only hovers, only in the viewer and scene video (not in the data), and at `GUARD_POS` it sits just inside the zone edge, so an intruder could bump into it there. Real fix: a defender that intercepts using the predictions, and add it to the data flights.
