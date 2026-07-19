\# Merged Trajectory Dataset — Report (Sakshi, Week 3 Fri)



\## Final counts

\- Total unique tasks merged: \*\*538\*\*

\- Passing: \*\*279 (51.9%)\*\*

\- Sources merged: Yeshita 7B (179), Faiza 14B (179), Sakshi API/local (180)

\- No task ID overlaps found across sources — clean merge, no dedup conflicts.



\## Notes for the team

\- This merged file supersedes the earlier 358-task version (which didn't

&#x20; include Sakshi's batch). Faiza's Week 5 training config uses the earlier

&#x20; 174-example split and is proceeding as planned — this update is available

&#x20; for a later "collect more data" decision if checkpoint evals show it's needed.

\- Sakshi's batch (`trajectories\_sakshi\_api.json`) started on Groq/Gemini API

&#x20; teachers but hit rate limits partway (Groq daily token cap, Gemini free

&#x20; tier not enabled on this account) — remaining \~90 tasks were completed

&#x20; using the local Ollama 7B model instead. This is reflected in a lower

&#x20; pass rate for that portion of the batch; worth noting as a real-world

&#x20; teacher-quality data point alongside Faiza's 7B vs 14B comparison.

\- All 538 trajectories pass schema validation (`datasets/schema.py`), zero

&#x20; invalid records.



\## Files

\- `datasets/collector.py` — merge script

\- `datasets/schema.py` — validator

\- `datasets/raw/trajectories\_merged.json` — final merged output

