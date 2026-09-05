import Foundation
import MLX
import MLXLMCommon

/// The module's one prefill mechanic, exposed without a store: consume `tokens[start...]` into
/// `cache` in pieces, checking `shouldPause` between pieces. A consumer that generates from a
/// plain prompt uses it so a cancel is honoured between pieces rather than after the whole
/// prompt; the coordinator's warm and capture paths use it beneath their block-aligned chunks.
///
/// Per piece: `model.prepare` on the piece (the library's windowed prefill), the trailing tokens
/// the iterator would otherwise sample from pushed through the model, then `eval` — so the cache
/// covers every token of the piece, attention and hybrid caches alike. Returns the offset reached:
/// `tokens.count` when done; less when `shouldPause` fired at that boundary, or when `prepare`
/// threw, in which case the cache covers exactly `reached` tokens and the caller decides.
/// `shouldPause` is consulted only after the first piece from `start`, so a resumed prefill
/// always makes progress.
///
/// Call inside `ModelContainer.perform`: the cache and the model are that context's.
public enum ChunkedPrefill {
    @discardableResult
    public static func run(
        _ tokens: [Int],
        into cache: [KVCache],
        from start: Int = 0,
        model: any LanguageModel,
        pieceSize: Int,
        windowSize: Int? = nil,
        shouldPause: () -> Bool = { false }
    ) -> Int {
        let piece = max(pieceSize, 1)
        var offset = start
        while offset < tokens.count {
            if offset > start, shouldPause() { return offset }
            let end = min(offset + piece, tokens.count)
            let input = LMInput(tokens: MLXArray(Array(tokens[offset ..< end])))
            guard let result = try? model.prepare(input, cache: cache, windowSize: windowSize ?? piece) else {
                return offset
            }
            if case let .tokens(remaining) = result {
                _ = model(remaining[text: .newAxis], cache: cache, state: nil)
            }
            eval(cache.flatMap { $0.state })
            offset = end
        }
        return offset
    }
}
