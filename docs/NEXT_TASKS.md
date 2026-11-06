# NEXT_TASKS.md

Day 4 (collaborative multi-agent intelligence) is complete — see
`docs/PROJECT_STATE.md` and `docs/CHANGELOG.md` "Session 5". No Day 5
was specified in this project's requirements.

## Candidate future improvements (not scheduled, not started)

- Extend Agent Agreement to consider equipment/cost/carbon agents with
  a documented (non-double-counting) contribution, not just
  forecast+anomaly.
- Cross-incident correlation (linking incidents across series/time),
  currently each series' incident is independent.
- A schedule model, so equipment-agent schedule-deviation claims could
  ever be genuinely confirmed rather than always "not confirmed by
  available evidence."
- Per-series MIN_ROWS_FOR_FORECAST tuned per role/sampling interval
  rather than one flat constant.
- LONG-format quality-scoring/cleaning refinements noted in earlier
  CHANGELOG sessions (mapping coverage cosmetics, duplicate-timestamp
  edge cases).
