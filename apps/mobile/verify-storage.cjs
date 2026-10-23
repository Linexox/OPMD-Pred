// Reproduce overlapping native saves without changing Android or patient files.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { execFileSync } = require('node:child_process');
const { randomUUID } = require('node:crypto');
const ts = require('typescript');

const baseline = process.argv.includes('--baseline');
const source = baseline
  ? execFileSync('git', ['show', 'HEAD:apps/mobile/src/storage.ts'], { cwd: __dirname, encoding: 'utf8' })
  : fs.readFileSync(path.join(__dirname, 'src/storage.ts'), 'utf8');

function harness() {
  const files = new Map();
  let failWrite = false, failRename = false;
  const pause = () => new Promise(resolve => setImmediate(resolve));
  const filesystem = {
    async writeFile({ path, data }) {
      await pause();
      if (failWrite) { failWrite = false; throw new Error('simulated write failure'); }
      files.set(path, data);
    },
    async rename({ from, to }) {
      await pause();
      if (failRename) { failRename = false; throw new Error('simulated rename failure'); }
      if (!files.has(from)) throw new Error(`rename: ${from} does not exist`);
      files.set(to, files.get(from)); files.delete(from);
    },
    async readFile({ path }) {
      if (!files.has(path)) throw new Error('missing file');
      return { data: files.get(path) };
    },
    async deleteFile({ path }) { files.delete(path); },
  };
  const module = { exports: {} };
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
  vm.runInNewContext(code, {
    module, exports: module.exports, crypto: { randomUUID },
    require(name) {
      if (name === '@capacitor/core') return { Capacitor: { isNativePlatform: () => true }, registerPlugin: () => ({}) };
      if (name === '@capacitor/filesystem') return { Filesystem: filesystem, Directory: { Data: 'DATA' }, Encoding: { UTF8: 'utf8' } };
      throw new Error('Unexpected dependency: ' + name);
    },
  });
  return { store: module.exports, files, failWrite: () => { failWrite = true; }, failRename: () => { failRename = true; } };
}

(async () => {
  const first = harness();
  const saves = await Promise.allSettled(Array.from({ length: 25 }, (_, revision) => first.store.persist('cache', { revision })));
  const rejected = saves.filter(result => result.status === 'rejected').length;
  if (baseline) {
    assert.ok(rejected > 0, 'The original implementation should reproduce the rename collision');
    console.log(JSON.stringify({ baseline: true, concurrent_saves: 25, rejected, collision_reproduced: true }));
    return;
  }
  assert.equal(rejected, 0);
  assert.equal((await first.store.restore('cache')).revision, 24);

  const snapshot = { revision: 25 };
  const saved = first.store.persist('cache', snapshot);
  snapshot.revision = 99;
  await saved;
  assert.equal((await first.store.restore('cache')).revision, 25);

  await Promise.all([first.store.persist('drafts', { draft: 'kept' }), first.store.persist('active-draft', { pending: true })]);
  assert.equal((await first.store.restore('drafts')).draft, 'kept');
  assert.equal((await first.store.restore('active-draft')).pending, true);

  const recovery = harness();
  await recovery.store.persist('cache', { revision: 1 });
  recovery.failWrite();
  await assert.rejects(recovery.store.persist('cache', { revision: 2 }));
  assert.equal((await recovery.store.restore('cache')).revision, 1);
  recovery.failRename();
  const failed = recovery.store.persist('cache', { revision: 3 });
  const retry = recovery.store.persist('cache', { revision: 4 });
  assert.deepEqual((await Promise.allSettled([failed, retry])).map(result => result.status), ['rejected', 'fulfilled']);
  assert.equal((await recovery.store.restore('cache')).revision, 4);
  assert.equal([...recovery.files.keys()].filter(name => name.endsWith('.tmp')).length, 0);
  console.log(JSON.stringify({ concurrent_saves: 25, rejected: 0, latest_snapshot: true, separate_keys: true, write_failure_preserves_old_file: true, queue_recovers_after_failure: true, failed_temp_files_cleaned: true }));
})().catch(error => { console.error(error); process.exitCode = 1; });
