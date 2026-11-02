// Provision dedicated fixtures directly; never change application auth behavior.
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import crypto from 'node:crypto';
const root = path.resolve(import.meta.dirname, '..');
const require = createRequire(path.join(root, 'backend-node/package.json'));
require('dotenv').config({ path: path.join(root, 'backend-node/.env'), quiet: true });
const bcrypt = require('bcryptjs');
const { db, newId, closeDb } = await import(pathToFileURL(path.join(root, 'backend-node/db.js')));
const file = path.join(import.meta.dirname, 'accounts-capacity.json');
try {
  if (fs.existsSync(file)) throw new Error('Fixture file already exists; refusing overwrite');
  const tag = crypto.randomBytes(8).toString('hex');
  const accounts = Array.from({ length: 150 }, (_, i) => ({
    email: `load-${tag}-${i}@example.invalid`, password: crypto.randomBytes(24).toString('hex'),
  }));
  fs.writeFileSync(file, JSON.stringify(accounts), { flag: 'wx', mode: 0o600 });
  for (const account of accounts) {
    await db.query(`INSERT INTO users (id,name,email,password,terms_accepted_at,terms_version,credits)
      VALUES ($1,$2,$3,$4,now(),$5,5)`,
      [newId(), 'Capacity test', account.email, await bcrypt.hash(account.password, 10), '2026-09-25']);
  }
  console.log('Created 150 dedicated capacity accounts; credentials are gitignored.');
} finally { await closeDb(); }
