#!/usr/bin/env bash
set -euo pipefail

DB_NAME="${CV_AUTO_TRAINER_PG_DB:-cv_auto_trainer}"
DB_USER="${CV_AUTO_TRAINER_PG_USER:-cv_auto_trainer}"
DB_PASSWORD="${CV_AUTO_TRAINER_PG_PASSWORD:-change-me}"

echo "Initializing PostgreSQL database for CV Auto Trainer..."
echo "  database: $DB_NAME"
echo "  user:     $DB_USER"

psql postgres <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${DB_USER}') THEN
    CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASSWORD}';
  END IF;
END
\$\$;
SELECT 'CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}')\gexec
ALTER DATABASE ${DB_NAME} OWNER TO ${DB_USER};
SQL

cat <<EOF

PostgreSQL is ready.

Use this environment variable before starting the backend:

export CV_AUTO_TRAINER_DB_URL=postgresql://${DB_USER}:${DB_PASSWORD}@127.0.0.1:5432/${DB_NAME}

Tables are created automatically by the backend on startup.
EOF
