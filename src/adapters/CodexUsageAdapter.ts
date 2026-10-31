import path from "node:path";
import { promises as fs } from "node:fs";
import type { AdapterImportResult, ImportIssue, SourceKind, SourceMeta, TokenBreakdown, UsageProvider, UsageRecord } from "../domain/types";
import { JsonUsageAdapter, jsonUsageParserVersion } from "./JsonUsageAdapter";
import type { UsageFileParseOutcome, UsageFileListResult } from "./JsonUsageAdapter";
import { streamJsonlLines, tailMatches } from "./jsonlStream";

const modelContextPaths = ["model", "message.model", "response.model", "payload.model", "payload.collaboration_mode.settings.model"];

type CodexFileContext = {
  sessionId: string;
  currentModel?: string;
  models: Set<string>;
  lastTotalUsage?: CodexTokenCounts;
  recentSnapshots?: Record<string, string>;
  previousSnapshot?: string;
  currentTurnId?: string;
  sawRequestUsage?: boolean;
};

/**
 * Resumable parse state for append-only Codex rollout files. Persisted in the
 * usage cache so a grown file only has its appended tail parsed. The context
 * carries everything line-order-dependent: the model set (for end-of-file
 * model backfill), the session id, and the running total_token_usage needed
 * for delta computation.
 */
export type CodexParseState = {
  parsedBytes: number;
  tailHash: string;
  lineCount: number;
  allowIncremental: boolean;
  context: {
    sessionId: string;
    currentModel?: string;
    models: string[];
    lastTotalUsage?: CodexTokenCounts;
    recentSnapshots?: Record<string, string>;
    previousSnapshot?: string;
    currentTurnId?: string;
  };
};

type PendingCodexRecord = Omit<UsageRecord, "model"> & {
  model?: string;
  mirrorKey?: string;
  requestUsage?: boolean;
};

type CodexTokenCounts = {
  input?: number;
  cachedInput?: number;
  output?: number;
};

export class CodexUsageAdapter extends JsonUsageAdapter {
  public readonly provider = "codex" as const;

  /** Include the standard sibling archive while keeping custom paths scoped. */
  public override async listUsageFiles(options?: { usagePath?: string }): Promise<UsageFileListResult> {
    const result = await super.listUsageFiles(options);
    const sourcePath = result.sourcePath;
    if (sourcePath && path.basename(path.resolve(sourcePath)).toLowerCase() === "sessions") {
      const archivePath = path.join(path.dirname(path.resolve(sourcePath)), "archived_sessions");
      try {
        const stat = await fs.lstat(archivePath);
        if (stat.isDirectory() && !stat.isSymbolicLink()) {
          const archive = await super.listUsageFiles({ usagePath: archivePath });
          result.files.push(...archive.files);
          result.errors.push(...archive.errors);
          result.warnings.push(...archive.warnings.filter((warning) => warning.code !== "empty_directory"));
          result.sourceMeta.push(...archive.sourceMeta);
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
          result.errors.push(issue("error", "directory_unreadable", errorMessage(error), archivePath, this.provider));
        }
      }
    }
    // A rollout can exist in both folders or as both compressed and plain
    // copies. Prefer the active folder, then its plain copy, once per session.
    const unique = new Map<string, UsageFileListResult["files"][number]>();
    for (const file of result.files) {
      const name = path.basename(file.filePath).replace(/\.zst$/i, "");
      const sessionId = /^rollout-.*-([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\.jsonl$/i.exec(name)?.[1];
      const key = sessionId?.toLowerCase() ?? path.resolve(file.filePath).replace(/\.zst$/i, "");
      const previous = unique.get(key);
      const previousIsActive = previous && sourcePath && isWithin(sourcePath, previous.filePath);
      const fileIsActive = sourcePath && isWithin(sourcePath, file.filePath);
      if (!previous || (!previousIsActive && fileIsActive) ||
          (previousIsActive === fileIsActive && previous.filePath.endsWith(".zst") && !file.filePath.endsWith(".zst"))) {
        unique.set(key, file);
      }
    }
    result.files = [...unique.values()];
    if (result.files.length > 10_000) {
      result.files = result.files.slice(0, 10_000);
      result.warnings.push(issue("warning", "source_file_limit", "Only the first 10000 Codex usage files were imported.", sourcePath, this.provider));
    }
    if (result.files.length > 0) {
      result.warnings = result.warnings.filter((warning) => warning.code !== "empty_directory");
    }
    return result;
  }

  protected override isUsageFile(name: string): boolean {
    return /\.jsonl\.zst$/i.test(name) || super.isUsageFile(name);
  }

  protected override async readFile(filePath: string, result: AdapterImportResult, readAt: string): Promise<void> {
    if (sourceKind(filePath) !== "jsonl") {
      await super.readFile(filePath, result, readAt);
      return;
    }
    await this.parseJsonl(filePath, result, readAt, undefined);
  }

  public override async importUsageFileWithState(filePath: string, prior?: unknown, readAt = new Date().toISOString()): Promise<UsageFileParseOutcome> {
    if (sourceKind(filePath) !== "jsonl") {
      return { result: await this.importUsageFile(filePath, readAt) };
    }

    if (filePath.toLowerCase().endsWith(".jsonl.zst")) {
      // Compressed byte positions cannot resume decompressed line offsets.
      return { result: await this.importUsageFile(filePath, readAt) };
    }
    const resume = validateCodexParseState(prior);
    if (resume && resume.allowIncremental && (await tailMatches(filePath, resume.parsedBytes, resume.tailHash))) {
      const appendedResult = this.emptyResult();
      const state = await this.parseJsonl(filePath, appendedResult, readAt, resume);
      if (state) {
        return { result: appendedResult, state, appended: true };
      }
      // The appended segment invalidated the resume context (e.g. a new model
      // appeared, changing the historical backfill); fall through to a full
      // parse with a fresh result.
    }

    const fullResult = this.emptyResult();
    const state = await this.parseJsonl(filePath, fullResult, readAt, undefined);
    return { result: fullResult, state };
  }

  private emptyResult(): AdapterImportResult {
    return { provider: this.provider, records: [], warnings: [], errors: [], sourceMeta: [] };
  }

  /**
   * Parses a rollout file, either fully (resume undefined) or from the byte
   * offset captured in `resume`. Returns the next parse state, or undefined
   * when a resumed parse turned out to be invalid and the caller must reparse
   * the whole file.
   */
  private async parseJsonl(
    filePath: string,
    result: AdapterImportResult,
    readAt: string,
    resume: CodexParseState | undefined,
  ): Promise<CodexParseState | undefined> {
    const meta = sourceMeta(filePath, "jsonl", readAt);
    result.sourceMeta.push(meta);
    const context: CodexFileContext = resume
      ? {
          sessionId: resume.context.sessionId,
          currentModel: resume.context.currentModel,
          models: new Set(resume.context.models),
          lastTotalUsage: resume.context.lastTotalUsage ? { ...resume.context.lastTotalUsage } : undefined,
          recentSnapshots: { ...resume.context.recentSnapshots },
          previousSnapshot: resume.context.previousSnapshot,
          currentTurnId: resume.context.currentTurnId,
        }
      : {
          sessionId: path.basename(filePath.replace(/\.zst$/i, ""), ".jsonl"),
          models: new Set(),
        };
    const records: PendingCodexRecord[] = [];
    const handleLine = (line: string, lineNumber: number): void => {
      if (!line.trim() || isPrefilteredCodexLine(line)) {
        return;
      }
      const row = parseJsonLine(line, filePath, lineNumber, result, this.provider);
      if (row === undefined) {
        return;
      }
      this.addCodexRecord(row, filePath, meta, result, lineNumber, context, records);
    };

    let stream;
    try {
      const before = /\.jsonl\.zst$/i.test(filePath) ? await fs.stat(filePath) : undefined;
      stream = await streamJsonlLines(filePath, handleLine, {
        start: resume?.parsedBytes ?? 0,
        lineNumberBase: resume?.lineCount ?? 0,
      });
      if (before) {
        const after = await fs.stat(filePath);
        if (before.size !== after.size || before.mtimeMs !== after.mtimeMs) {
          result.warnings.push(issue("warning", "file_transient", "Compressed session changed while reading; skipped for this refresh.", filePath, this.provider));
          return undefined;
        }
      }
    } catch (error) {
      result.errors.push(issue("error", "file_unreadable", errorMessage(error), filePath, this.provider));
      return undefined;
    }

    // Parity with the previous reader, which emitted the unterminated last
    // line: process it in both full and resume mode so a crashed session's
    // final record still counts. A processed tail poisons incremental resume
    // (the completed line would be parsed a second time), so it disables it —
    // the next change then takes the idempotent full-reparse path.
    let processedPartialTail = false;
    if (stream.partialTail !== undefined) {
      handleLine(stream.partialTail, (resume?.lineCount ?? 0) + stream.lineCount + 1);
      processedPartialTail = true;
    }

    if (!resume && !stream.sawContent) {
      result.warnings.push(issue("warning", "empty_file", "Usage file is empty.", filePath, this.provider));
      return {
        parsedBytes: stream.parsedBytes,
        tailHash: stream.tailHash,
        lineCount: stream.lineCount,
        allowIncremental: true,
        context: serializeCodexContext(context),
      };
    }

    if (resume && (context.sawRequestUsage || !modelsUnchanged(resume.context.models, context.models))) {
      // Any change to the model set (including 0 -> 1) alters the model
      // backfill applied to records parsed earlier, so the whole file must be
      // reparsed for those records to come out right.
      return undefined;
    }

    const inferredModel = context.models.size === 1 ? [...context.models][0] : undefined;
    // Newer rollouts may append request usage alongside the legacy token-count
    // events. Match one mirror per turn and token tuple, preserving unmatched
    // request-only records and repeated equal-sized requests.
    const legacyMirrors = new Map<string, number>();
    for (const record of records) {
      if (record.mirrorKey && !record.requestUsage) {
        legacyMirrors.set(record.mirrorKey, (legacyMirrors.get(record.mirrorKey) ?? 0) + 1);
      }
    }
    const distinctRecords = records.filter((record) => {
      const copies = record.requestUsage && record.mirrorKey ? legacyMirrors.get(record.mirrorKey) ?? 0 : 0;
      if (copies > 0) {
        legacyMirrors.set(record.mirrorKey!, copies - 1);
        return false;
      }
      return true;
    });
    result.records.push(
      ...distinctRecords.map(({ mirrorKey, requestUsage, ...record }) => ({
        ...record,
        model: record.model ?? inferredModel,
        sessionId: record.sessionId ?? context.sessionId,
      })),
    );

    return {
      parsedBytes: stream.parsedBytes,
      tailHash: stream.tailHash,
      lineCount: (resume?.lineCount ?? 0) + stream.lineCount,
      allowIncremental: !processedPartialTail && !context.sawRequestUsage,
      context: serializeCodexContext(context),
    };
  }

  private addCodexRecord(
    row: unknown,
    filePath: string,
    meta: SourceMeta,
    result: AdapterImportResult,
    line: number,
    context: CodexFileContext,
    records: PendingCodexRecord[],
  ): void {
    const object = asObject(row);
    if (!object) {
      result.warnings.push(issue("warning", "record_not_object", "Usage record is not an object.", filePath, this.provider, line));
      return;
    }

    this.updateContext(object, context);
    const payload = asObject(object["payload"]);
    const requestUsage = object["type"] === "token_usage_record";
    if (requestUsage) {
      context.sawRequestUsage = true;
    }
    const usage = this.usageForRecord(object, payload, context);
    if (object["type"] === "event_msg" && payload?.["type"] === "token_count" && !usage) {
      const info = payload["info"];
      if (info === null || info === undefined) {
        // Rate-limit-only heartbeats ship `info: null` by design; they carry no token usage and are not import issues.
        return;
      }
      result.warnings.push(issue("warning", "missing_tokens", "Codex token_count event has info but no token usage payload.", filePath, this.provider, line));
      return;
    }
    if (!usage) {
      return;
    }

    const tokens = tokensFromCodexCounts(usage);
    if (!hasNonZeroTokenUsage(tokens)) {
      return;
    }

    records.push({
      requestUsage,
      mirrorKey: (requestUsage ? firstString(payload ?? {}, ["turn_id"]) : context.currentTurnId)
        ? JSON.stringify([requestUsage ? firstString(payload ?? {}, ["turn_id"]) : context.currentTurnId,
            usage.input, usage.cachedInput, usage.output])
        : undefined,
      provider: this.provider,
      model: firstString(object, modelContextPaths) ?? context.currentModel,
      sessionId: firstString(object, ["sessionId", "session_id", "conversationId", "conversation_id", "uuid", "payload.id", "payload.session_id"]) ?? context.sessionId,
      startedAt: normalizeIso(firstString(object, ["startedAt", "started_at", "timestamp", "createdAt", "created_at"])),
      endedAt: normalizeIso(firstString(object, ["endedAt", "ended_at", "completedAt", "completed_at"])),
      observedAt: normalizeIso(firstString(object, ["observedAt", "observed_at", "timestamp", "createdAt", "created_at"])) ?? meta.readAt,
      tokens,
      source: meta,
      raw: row,
    });
  }

  private usageForRecord(object: Record<string, unknown>, payload: Record<string, unknown> | undefined, context: CodexFileContext): CodexTokenCounts | undefined {
    if (object["type"] === "event_msg" && payload) {
      const totalUsage = countsFromCodexUsage(asObject(valueAt(payload, "info.total_token_usage")));
      const lastUsage = countsFromCodexUsage(asObject(valueAt(payload, "info.last_token_usage")));
      if (totalUsage) {
        const signature = JSON.stringify([totalUsage, lastUsage]);
        const source = firstString(payload, ["rate_limits.limit_id", "info.rate_limits.limit_id"]) ?? "default";
        const duplicate = context.previousSnapshot === signature || context.recentSnapshots?.[source] === signature;
        context.recentSnapshots ??= {};
        context.recentSnapshots[source] = signature;
        context.previousSnapshot = signature;
        const keys = Object.keys(context.recentSnapshots);
        if (keys.length > 32) {
          delete context.recentSnapshots[keys[0]!];
        }
        const delta = deltaCounts(totalUsage, context.lastTotalUsage);
        context.lastTotalUsage = {
          input: Math.max(totalUsage.input ?? 0, context.lastTotalUsage?.input ?? 0),
          cachedInput: Math.max(totalUsage.cachedInput ?? 0, context.lastTotalUsage?.cachedInput ?? 0),
          output: Math.max(totalUsage.output ?? 0, context.lastTotalUsage?.output ?? 0),
        };
        // Request-local counts avoid re-attributing inherited session totals.
        // Repeated snapshots are status updates, not additional requests.
        return duplicate ? { input: 0, cachedInput: 0, output: 0 } : lastUsage ?? delta;
      }
      if (lastUsage) {
        return lastUsage;
      }
      return undefined;
    }

    const genericUsage = asObject(object["usage"]) ?? asObject(valueAt(object, "message.usage")) ?? asObject(valueAt(object, "response.usage")) ?? asObject(valueAt(object, "payload.usage"));
    return countsFromCodexUsage(genericUsage);
  }

  private updateContext(object: Record<string, unknown>, context: CodexFileContext): void {
    if (object["type"] === "turn_context") {
      context.currentTurnId = firstString(object, ["payload.turn_id"]);
    }
    const model = firstString(object, modelContextPaths);
    if (model) {
      context.currentModel = model;
      context.models.add(model);
    }
    const sessionId = firstString(object, ["sessionId", "session_id", "conversationId", "conversation_id", "uuid", "payload.id", "payload.session_id"]);
    if (sessionId) {
      context.sessionId = sessionId;
    }
  }
}

// Rollout bulk lines (assistant output, tool transcripts) carry no usage or
// context fields and account for ~90% of file bytes. Lines whose top-level
// type marks them as bulk are skipped before JSON.parse unless they mention a
// field the parser would actually read (conservative substring guard).
const prefilterSkippableTypes = new Set(["compacted", "response_item"]);
const prefilterContextPattern = /"(model|usage|sessionId|session_id|conversationId|conversation_id|uuid)"\s*:/;

function isPrefilteredCodexLine(line: string): boolean {
  const typeIndex = line.indexOf('"type":"');
  if (typeIndex === -1 || typeIndex > 80) {
    return false;
  }
  const typeEnd = line.indexOf('"', typeIndex + 8);
  if (typeEnd === -1) {
    return false;
  }
  const type = line.slice(typeIndex + 8, typeEnd);
  if (!prefilterSkippableTypes.has(type)) {
    return false;
  }
  return !prefilterContextPattern.test(line);
}

function serializeCodexContext(context: CodexFileContext): CodexParseState["context"] {
  return {
    sessionId: context.sessionId,
    currentModel: context.currentModel,
    models: [...context.models].sort(),
    lastTotalUsage: context.lastTotalUsage ? { ...context.lastTotalUsage } : undefined,
    recentSnapshots: context.recentSnapshots ? { ...context.recentSnapshots } : undefined,
    previousSnapshot: context.previousSnapshot,
    currentTurnId: context.currentTurnId,
  };
}

function modelsUnchanged(prior: string[], current: Set<string>): boolean {
  if (prior.length !== current.size) {
    return false;
  }
  return prior.every((model) => current.has(model));
}

function validateCodexParseState(value: unknown): CodexParseState | undefined {
  if (typeof value !== "object" || value === null) {
    return undefined;
  }
  const candidate = value as Partial<CodexParseState>;
  const context = candidate.context;
  if (
    !Number.isSafeInteger(candidate.parsedBytes) ||
    (candidate.parsedBytes as number) < 0 ||
    typeof candidate.tailHash !== "string" ||
    !Number.isSafeInteger(candidate.lineCount) ||
    (candidate.lineCount as number) < 0 ||
    typeof candidate.allowIncremental !== "boolean" ||
    typeof context !== "object" ||
    context === null ||
    typeof context.sessionId !== "string" ||
    (context.currentModel !== undefined && typeof context.currentModel !== "string") ||
    !Array.isArray(context.models) ||
    context.models.some((model) => typeof model !== "string") ||
    !validOptionalCounts(context.lastTotalUsage) ||
    (context.previousSnapshot !== undefined && typeof context.previousSnapshot !== "string") ||
    (context.currentTurnId !== undefined && typeof context.currentTurnId !== "string") ||
    (context.recentSnapshots !== undefined && (typeof context.recentSnapshots !== "object" ||
      context.recentSnapshots === null || Array.isArray(context.recentSnapshots) ||
      Object.keys(context.recentSnapshots).length > 32 ||
      Object.values(context.recentSnapshots).some((value) => typeof value !== "string")))
  ) {
    return undefined;
  }
  return candidate as CodexParseState;
}

function validOptionalCounts(value: unknown): boolean {
  if (value === undefined) {
    return true;
  }
  if (typeof value !== "object" || value === null) {
    return false;
  }
  return (["input", "cachedInput", "output"] as const).every((key) => {
    const count = (value as Record<string, unknown>)[key];
    return count === undefined || (typeof count === "number" && Number.isFinite(count));
  });
}

function parseJsonLine(
  line: string,
  filePath: string,
  lineNumber: number,
  result: AdapterImportResult,
  provider: UsageProvider,
): unknown | undefined {
  const trimmed = line.trim();
  try {
    return JSON.parse(trimmed) as unknown;
  } catch (error) {
    if (trimmed.startsWith("\\{")) {
      try {
        return JSON.parse(trimmed.slice(1)) as unknown;
      } catch {
        // Fall through to the original parse error for the actionable diagnostic.
      }
    }
    result.errors.push(issue("error", "malformed_jsonl", errorMessage(error), filePath, provider, lineNumber));
    return undefined;
  }
}

function countsFromCodexUsage(usage: Record<string, unknown> | undefined): CodexTokenCounts | undefined {
  if (!usage) {
    return undefined;
  }
  const rawInput = numberValue(usage["input_tokens"] ?? usage["inputTokens"] ?? usage["input"]);
  const cachedInput = numberValue(usage["cached_input_tokens"] ?? usage["cachedInputTokens"] ?? usage["cachedInput"] ?? usage["cached_input"]);
  const output = numberValue(usage["output_tokens"] ?? usage["outputTokens"] ?? usage["output"]);
  if (rawInput === undefined && cachedInput === undefined && output === undefined) {
    return undefined;
  }
  return {
    input: rawInput,
    cachedInput,
    output,
  };
}

function deltaCounts(current: CodexTokenCounts, previous: CodexTokenCounts | undefined): CodexTokenCounts {
  if (!previous) {
    return current;
  }
  return {
    input: deltaNumber(current.input, previous.input),
    cachedInput: deltaNumber(current.cachedInput, previous.cachedInput),
    output: deltaNumber(current.output, previous.output),
  };
}

function deltaNumber(current: number | undefined, previous: number | undefined): number | undefined {
  if (current === undefined) {
    return undefined;
  }
  return Math.max(0, current - (previous ?? 0));
}

function tokensFromCodexCounts(counts: CodexTokenCounts): TokenBreakdown {
  const tokens: TokenBreakdown = {};
  if (counts.input !== undefined) {
    tokens.input = Math.max(0, counts.input - (counts.cachedInput ?? 0));
  }
  if (counts.cachedInput !== undefined) {
    tokens.cachedInput = counts.cachedInput;
  }
  if (counts.output !== undefined) {
    tokens.output = counts.output;
  }
  return tokens;
}

function hasNonZeroTokenUsage(tokens: TokenBreakdown): boolean {
  return Object.values(tokens).some((count) => typeof count === "number" && count > 0);
}

function sourceKind(filePath: string): SourceKind {
  if (/\.jsonl\.zst$/i.test(filePath)) {
    return "jsonl";
  }
  const extension = path.extname(filePath).toLowerCase();
  if (extension === ".jsonl") {
    return "jsonl";
  }
  if (extension === ".json") {
    return "json";
  }
  return "unknown";
}

function sourceMeta(sourcePath: string, sourceKindValue: SourceKind, readAt: string): SourceMeta {
  return {
    sourcePath,
    sourceKind: sourceKindValue,
    schemaVersion: "local-usage-v2",
    parserVersion: jsonUsageParserVersion,
    readAt,
  };
}

function issue(
  severity: "warning" | "error",
  code: string,
  message: string,
  sourcePath?: string,
  provider?: UsageProvider,
  line?: number,
): ImportIssue {
  return { severity, code, message, sourcePath, provider, line };
}

function asObject(value: unknown): Record<string, unknown> | undefined {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return undefined;
}

function valueAt(object: Record<string, unknown>, pathValue: string): unknown {
  let current: unknown = object;
  for (const segment of pathValue.split(".")) {
    const currentObject = asObject(current);
    if (!currentObject) {
      return undefined;
    }
    current = currentObject[segment];
  }
  return current;
}

function firstString(object: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = key.includes(".") ? valueAt(object, key) : object[key];
    if (typeof value === "string" && value.trim()) {
      return value;
    }
  }
  return undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizeIso(value: string | undefined): string | undefined {
  if (!value) {
    return undefined;
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return undefined;
  }
  return parsed.toISOString();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isWithin(directory: string, filePath: string): boolean {
  const relative = path.relative(path.resolve(directory), path.resolve(filePath));
  return relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative);
}
