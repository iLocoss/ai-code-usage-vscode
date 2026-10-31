import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { test } from "node:test";
import { zstdCompressSync } from "node:zlib";
import { CodexUsageAdapter } from "../src/adapters/CodexUsageAdapter";
import { CachedUsageImporter } from "../src/services/CachedUsageImporter";
import { TimeRangeService } from "../src/services/TimeRangeService";

const session = ["00000000", "0000", "4000", "8000", "000000000001"].join("-");
const name = "rollout-2026-04-30T08-00-00-" + session + ".jsonl";
const row = JSON.stringify({
  model: "gpt-6-sol", sessionId: session, timestamp: "2026-04-30T08:01:00.000Z",
  usage: { input_tokens: 100, cached_input_tokens: 80, output_tokens: 5 },
});

test("Codex uses request-local usage and ignores repeated snapshots across a cache resume", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "usage-counts-"));
  try {
    const file = path.join(dir, name);
    const event = (time: string, total: number, last: number) => JSON.stringify({
      timestamp: time, type: "event_msg", payload: {
        type: "token_count", info: {
          total_token_usage: { input_tokens: total, cached_input_tokens: total / 2, output_tokens: total / 10 },
          last_token_usage: { input_tokens: last, cached_input_tokens: last / 2, output_tokens: last / 10 },
        },
      },
    });
    const first = event("2026-04-30T08:00:00Z", 10000, 100);
    const adapter = new CodexUsageAdapter();
    await writeFile(file, first + "\n");
    const a = await adapter.importUsageFileWithState(file);
    assert.equal(a.result.records[0]?.tokens.input, 50);
    assert.equal(a.result.records[0]?.tokens.output, 10);
    await writeFile(file, first + "\n" + event("2026-04-30T08:01:00Z", 10200, 200) + "\n", { flag: "a" });
    const b = await adapter.importUsageFileWithState(file, a.state);
    assert.equal(b.appended, true);
    assert.equal(b.result.records.length, 1);
    assert.equal(b.result.records[0]?.tokens.input, 100);
    assert.equal(b.result.records[0]?.tokens.output, 20);
    assert.equal(b.result.warnings.length, 0);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Codex compressed JSONL matches plain input, including UTF-8 and a partial final line", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "usage-zstd-"));
  try {
    const plain = path.join(dir, name);
    const compressed = plain + ".zst";
    const content = row + "\r\n" + row.replace(session, "fixture-é-二");
    await writeFile(plain, content);
    await writeFile(compressed, zstdCompressSync(Buffer.from(content)));
    const adapter = new CodexUsageAdapter();
    const a = await adapter.importUsageFile(plain);
    const b = await adapter.importUsageFileWithState(compressed, { parsedBytes: 99, allowIncremental: true });
    assert.equal(a.records.length, 2);
    assert.deepEqual(b.result.records.map(({ source, ...record }) => record), a.records.map(({ source, ...record }) => record));
    assert.equal(b.state, undefined);
    assert.equal(b.appended, undefined);
    assert.equal(b.result.errors.length, 0);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Codex retains request-only usage and removes only matching legacy mirrors in the same turn", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "usage-mirrors-"));
  try {
    const file = path.join(dir, name);
    const usage = { input_tokens: 100, cached_input_tokens: 80, output_tokens: 5 };
    const lines = [
      { type: "turn_context", payload: { model: "gpt-6-sol", turn_id: "fixture-turn" } },
      { timestamp: "2026-04-30T08:00:00Z", type: "event_msg", payload: {
        type: "token_count", info: { last_token_usage: usage, total_token_usage: usage },
      } },
      { timestamp: "2026-04-30T08:00:01Z", type: "token_usage_record", payload: { turn_id: "fixture-turn", usage } },
      // Equal usage on another turn is real work, not a duplicate.
      { timestamp: "2026-04-30T08:00:02Z", type: "token_usage_record", payload: { turn_id: "fixture-other", usage } },
    ].map((value) => JSON.stringify(value)).join("\n") + "\n";
    await writeFile(file, lines);
    const adapter = new CodexUsageAdapter();
    const result = await adapter.importUsageFileWithState(file);
    assert.equal(result.result.records.length, 2);
    assert.equal((result.state as { allowIncremental: boolean }).allowIncremental, false);
    assert.ok(result.result.records.every((record) => !("mirrorKey" in record) && !("requestUsage" in record)));
    // Request records arriving after a prior legacy-only parse force a full
    // parse, so cache appends cannot preserve both representations.
    await writeFile(file, lines.split("\n").slice(0, 2).join("\n") + "\n");
    const prior = await adapter.importUsageFileWithState(file);
    await writeFile(file, lines);
    const reread = await adapter.importUsageFileWithState(file, prior.state);
    assert.equal(reread.appended, undefined);
    assert.equal(reread.result.records.length, 2);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Codex handles concatenated compressed frames and an empty frame", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "usage-frames-"));
  try {
    const file = path.join(dir, name + ".zst");
    await writeFile(file, Buffer.concat([
      zstdCompressSync(Buffer.alloc(0)),
      zstdCompressSync(Buffer.from(row + "\n")),
      zstdCompressSync(Buffer.from(row + "\n")),
    ]));
    const result = await new CodexUsageAdapter().importUsageFile(file);
    assert.equal(result.records.length, 2);
    assert.equal(result.errors.length, 0);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Codex lists sibling archives and counts plain/compressed rollout copies only once", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "usage-zstd-"));
  try {
    const sessions = path.join(dir, "sessions");
    const archives = path.join(dir, "archived_sessions");
    await mkdir(sessions);
    await mkdir(archives);
    await writeFile(path.join(sessions, name), row + "\n");
    await writeFile(path.join(sessions, name + ".zst"), zstdCompressSync(Buffer.from(row + "\n")));
    await writeFile(path.join(archives, name + ".zst"), zstdCompressSync(Buffer.from(row + "\n")));
    const other = name.replace(/0001\.jsonl$/, "0002.jsonl");
    await writeFile(path.join(archives, other + ".zst"), zstdCompressSync(Buffer.from(row.replace(session, session.replace(/1$/, "2")) + "\n")));
    const listed = await new CodexUsageAdapter(sessions).listUsageFiles();
    assert.equal(listed.files.length, 2);
    assert.ok(listed.files.some((file) => file.filePath === path.join(sessions, name)));
    assert.ok(listed.files.some((file) => file.filePath === path.join(archives, other + ".zst")));
    const parsed = await new CodexUsageAdapter(sessions).importUsage();
    assert.equal(parsed.records.length, 2);
    assert.equal(parsed.errors.length, 0);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Codex reports corrupt compressed sessions without importing partial records", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "usage-zstd-"));
  try {
    const file = path.join(dir, name + ".zst");
    await writeFile(file, Buffer.from("not a zstd frame"));
    const result = await new CodexUsageAdapter().importUsageFile(file);
    assert.equal(result.records.length, 0);
    assert.equal(result.errors[0]?.code, "file_unreadable");
    const valid = zstdCompressSync(Buffer.from(row + "\n"));
    await writeFile(file, valid.subarray(0, valid.length - 3));
    const truncated = await new CodexUsageAdapter().importUsageFile(file);
    assert.equal(truncated.records.length, 0);
    assert.equal(truncated.errors[0]?.code, "file_unreadable");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Codex custom directories do not discover unrelated sibling archives", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "usage-zstd-"));
  try {
    const custom = path.join(dir, "custom");
    const archives = path.join(dir, "archived_sessions");
    await mkdir(custom);
    await mkdir(archives);
    await writeFile(path.join(custom, name + ".zst"), zstdCompressSync(Buffer.from(row + "\n")));
    await writeFile(path.join(archives, "other.jsonl"), row + "\n");
    const result = await new CodexUsageAdapter(custom).importUsage();
    assert.equal(result.records.length, 1);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("Codex compressed cache warm reads, rebuilds and compression transitions remain idempotent", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "usage-zstd-"));
  try {
    const sessions = path.join(dir, "sessions");
    await mkdir(sessions);
    const plain = path.join(sessions, name);
    await writeFile(plain, row + "\n");
    const cache = new CachedUsageImporter(path.join(dir, "cache"));
    const range = new TimeRangeService().resolve("custom", { start: "2026-04-30", end: "2026-04-30" });
    const input = { sources: [{ provider: "codex" as const, sourcePath: sessions, adapter: new CodexUsageAdapter() }], range };
    assert.equal((await cache.loadForRange(input)).imports[0]?.records.length, 1);
    await writeFile(plain + ".zst", zstdCompressSync(await readFile(plain)));
    await rm(plain);
    for (const forceReparse of [false, false, true]) {
      const loaded = await cache.loadForRange({ ...input, forceReparse });
      assert.equal(loaded.imports[0]?.records.length, 1);
      assert.equal(loaded.imports[0]?.errors.length, 0);
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
