import { Decompress } from "fzstd";

/** Decode frames incrementally and require a complete final frame. */
export async function* decompressZstdChunks(source: AsyncIterable<Uint8Array>): AsyncGenerator<Buffer> {
  let output: Buffer[] = [];
  const decoder = new Decompress((chunk) => {
    if (chunk.length) {
      // The decoder reuses its sliding window after each callback.
      output.push(Buffer.from(chunk));
    }
  });
  for await (const chunk of source) {
    decoder.push(chunk);
    yield* output;
    output = [];
  }
  decoder.push(new Uint8Array(0), true);
  yield* output;
}
