# Testnet / Mainnet Isolation via Postgres Schemas

Radar runs a single centralized Postgres database that serves BOTH the testnet
and mainnet validator networks. The two networks are kept strictly isolated
by putting each network's tables in its own Postgres **schema** — `testnet`
and `mainnet` — rather than by a `network` tag column on shared tables or by
running separate databases.

## Why schemas?

- **Zero changes to existing SQL.** Every query in `shared/pg_*.py` uses
  unqualified table names (`SELECT ... FROM experiments`). Postgres resolves
  those through the connection's `search_path`, which we set to the network's
  schema on every connection borrowed from the pool. A forgotten
  `WHERE network = ...` cannot leak mainnet rows into a testnet result,
  because the other schema's tables are simply not in scope.
- **Per-schema sequences.** `SERIAL` columns (`round_context.id`,
  `code_components.id`, etc.) live inside the schema, so testnet and mainnet
  have independent ID spaces automatically.
- **Backup and drop are per-network.** `pg_dump --schema=mainnet` dumps only
  mainnet; `DROP SCHEMA testnet CASCADE` wipes testnet without touching
  mainnet.
- **Foreign keys stay intra-schema.** `parent_index INTEGER REFERENCES
  experiments(id)` resolves against `search_path` at `CREATE TABLE` time, so
  the FK correctly targets the current schema's `experiments` table.

## How the switch happens

1. `config.Config.NETWORK` reads `RADAR_NETWORK` (default `testnet`).
2. `database/neuron.py::_init_db` opens a **bare** `asyncpg.connect()` (NOT
   a pool connection — the pool pins `search_path` to the schema, which
   would make "CREATE SCHEMA" ask the schema to create itself) and calls
   `shared.pg_store.ensure_schema_exists(conn, Config.NETWORK)`.
3. It then calls `shared.pg_store.create_pg_pool(dsn, schema=Config.NETWORK,
   ...)`. Every connection this pool hands out runs
   `SET search_path TO "<schema>", public` in its init hook (after the
   JSON/JSONB codec registration).
4. `PgExperimentStore.init_schema()`, `PgProvenanceQuery.init_schema()`,
   `PgAccessLogger.init_schema()`, and the other `conn.execute(<DDL>)`
   calls in `_init_db` all run inside the correct schema because
   `search_path` is already set.

## Safety: the schema-name allowlist

Postgres identifiers cannot be parameterised in prepared statements, so the
schema name is spliced directly into SQL (`SET search_path TO "<schema>",
public`). The allowlist regex in `shared/pg_store.py::_validate_schema_name`
is the ONLY defense against SQL injection through this variable:

```
^[a-z][a-z0-9_]{0,62}$
```

Anything else raises `ValueError` at pool creation time. Never bypass this
helper.

## Operator cheat-sheet

### Check which schema a connection is using

```sql
SHOW search_path;
-- expected: "testnet", public   (or "mainnet", public)
```

### Inspect the other network manually

By default a `psql` session comes in on `search_path=public` (or whatever
the role's default is). To peek at testnet or mainnet data interactively:

```sql
-- see what schemas exist
\dn

-- temporarily switch for this session
SET search_path TO mainnet;
SELECT COUNT(*) FROM experiments;

-- or keep both schemas in scope for comparison
SET search_path TO testnet, public;
SELECT COUNT(*) AS testnet_rows FROM testnet.experiments;
SELECT COUNT(*) AS mainnet_rows FROM mainnet.experiments;
```

Fully-qualified names (`SELECT * FROM mainnet.experiments LIMIT 5`) always
work regardless of `search_path`.

### Back up just one network

```bash
# Dump only mainnet
pg_dump --schema=mainnet -Fc -f radar_mainnet.dump \
    "postgresql://user:pass@host:5432/radar"

# Dump only testnet
pg_dump --schema=testnet -Fc -f radar_testnet.dump \
    "postgresql://user:pass@host:5432/radar"
```

### Drop and rebuild testnet from scratch without touching mainnet

```sql
-- Destructive. Make sure you're connected to the right DB.
DROP SCHEMA testnet CASCADE;
```

Then restart `database/neuron.py` with `RADAR_NETWORK=testnet` — the next
boot will `CREATE SCHEMA IF NOT EXISTS "testnet"` and re-run the DDL, giving
you a fresh empty testnet while mainnet keeps running untouched.

### Restore from a dump

```bash
pg_restore -d "postgresql://user:pass@host:5432/radar" radar_testnet.dump
```

## FTS trigger guard

`shared/pg_schema.py::FTS_TRIGGER_DDL` uses
`pg_trigger` to check whether the FTS trigger has already been created
before re-creating it. The `tgname` column is unique **per relation**, not
globally, so the guard qualifies the lookup with
`tgrelid = 'experiments'::regclass`. That cast resolves through the current
`search_path`, so the check looks at the current schema's `experiments`
table only — if mainnet already has `experiments_search_trigger`, testnet
still (correctly) creates its own.

## Testing

The existing test suite (`tests/test_pg_store.py`, `tests/test_pg_provenance.py`,
etc.) uses the default `RADAR_NETWORK=testnet`, which matches historic
behaviour: tests create their tables in the `testnet` schema. If you want a
test run not to touch any real testnet data, point `TEST_PG_DSN` at a
throwaway database or set `RADAR_NETWORK` to a scratch schema name (must
match the allowlist regex, e.g. `ci_scratch`).

## What is NOT done here

- No `network` column is added to any table.
- No separate databases are created (one `radar` database holds both
  schemas).
- No existing SQL in `shared/pg_*.py` is modified to add schema qualifiers
  — every query stays unqualified and relies on `search_path`. That is the
  whole point of this design.
