# Schema migrations

This directory holds forward-only SQL migrations applied automatically at
database server startup by `shared/migrations.py`. Migrations are applied
per-schema: `testnet` and `mainnet` track their own history in
`<schema>.schema_migrations` and are advanced independently.

## File convention

```
NNN_short_description.sql
```

- `NNN` is a zero-padded 3-digit prefix (`001`, `002`, ...).
- Prefixes must be strictly sequential with no gaps.
- Description is lowercase ASCII, digits, and underscores only.
- Anything that doesn't match `^\d{3}_[a-z0-9_]+\.sql$` is rejected by the
  runner.

Examples:

- `001_add_priority.sql`
- `002_drop_legacy_trace.sql`
- `003_backfill_round_id.sql`

## Schema placeholder

The runner applies the same migration file to every managed schema.
Every schema qualifier must use the literal token `CURRENT_SCHEMA`, which
is replaced at apply time with the validated schema name (`testnet` or
`mainnet`). The replacement is a plain string substitution, so avoid
using the literal string `CURRENT_SCHEMA` for anything other than the
schema placeholder.

```sql
-- Good
ALTER TABLE CURRENT_SCHEMA.experiments ADD COLUMN IF NOT EXISTS priority INT DEFAULT 0;
CREATE INDEX IF NOT EXISTS experiments_priority_idx
  ON CURRENT_SCHEMA.experiments (priority);

-- Bad — hardcoded schema will only ever touch one environment
ALTER TABLE testnet.experiments ADD COLUMN priority INT;
```

## Idempotency

Each migration runs inside its own transaction and is recorded in
`<schema>.schema_migrations` only on success. Still, prefer idempotent
DDL (`CREATE ... IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`,
`DROP ... IF EXISTS`) so partial re-runs, aborted deploys, or manually
applied hotfixes don't wedge the migration runner.

## Immutability

Committed migrations are immutable. If a file's contents change after it
has been applied, the runner logs a loud warning but keeps going.
Editing a committed migration is almost always a mistake — write a new
one instead.

## When to add a migration

Any pull request that changes `shared/pg_schema.py` must ship a
companion `.sql` file here. CI enforces this. The only exception is a
brand-new table defined via `CREATE TABLE IF NOT EXISTS` that will be
bootstrapped via `init_schema()` on fresh databases — but you still need
a migration if existing deployments need the change applied.

## Destructive changes

Changes containing `DROP TABLE`, `DROP COLUMN`, or `ALTER COLUMN ... TYPE`
require the string `ALLOW_DESTRUCTIVE` in the pull request description.
CI blocks the PR otherwise.

## What not to put here

- No DSNs, passwords, connection strings, or secrets of any kind.
- No runtime data the app should own (seed through application code).
- No `CREATE SCHEMA` — schemas are created by
  `ensure_schema_exists()` before migrations run.
