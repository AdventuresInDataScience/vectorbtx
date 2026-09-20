# Developer documentation

Not user-facing. These record **how the library came to be what it is** — the design
argument, the corrections, and the environment it was built in. They read like working
notes because that is what they are.

| File | What it is |
|---|---|
| `PLAN.md` | The full design plan, section by section, with "Done when" criteria. The reference for *why* a decision was made. |
| `AUDIT-G.md` | Domain audit of the holding-cost model against broker documentation. Records where the original plan was wrong and what replaced it. |
| `AUDIT.md` | Implementation audit against the plan. |
| `FRONTEND-API.md` | The front-door API design note, written before it was built. |
| `CORRECTIONS.md` | Build log of silent-wrong-answer bugs found by executing the examples. The argument for keeping the harnesses in CI. |
| `SANDBOX-ARTEFACTS.md` | What in this repo exists only because the build environment had no network or compiler. Read this first when moving to a real dev machine. |

User documentation is one level up: `docs/API.md`, `docs/EXAMPLES.md`, and the `README.md`
at the repository root.
