import Foundation

/// Settings for an OCR inference run over PDFs/images via `Backend/ocr_infer.py`
/// (the Unlimited-OCR MLX model). Drives the OCR page in the app.
struct OCRConfig: Equatable {
    static let defaultModel = "baidu/Unlimited-OCR"
    static let defaultPrompt = "<image>\n<|grounding|>Convert the document to markdown."

    /// A single file (PDF/image) or a folder. Folders honor `recursive`.
    var inputPath = ""
    var recursive = false
    var model = Self.defaultModel
    /// Optional trained LoRA adapter (`adapters.safetensors`).
    var adapterPath = ""
    /// Where `.md` outputs go. Empty → the runner creates a run folder.
    var outputDir = ""
    var prompt = Self.defaultPrompt
    var maxTokens = 4000
    var dpi = 144
    /// 0 → all pages.
    var maxPages = 0
    /// 1.0 disables the penalty; > 1 curbs repetition on dense figure pages.
    var repetitionPenalty = 1.0
    var repetitionContextSize = 20

    /// Build the argv for `Backend/ocr_infer.py`. `resolvedOutputDir` is the
    /// directory the runner decided to write `.md` files into.
    func runArguments(resolvedOutputDir: String) -> [String] {
        var args = ["Backend/ocr_infer.py", inputPath.trimmingCharacters(in: .whitespacesAndNewlines)]
        if recursive { args.append("--recursive") }
        args += ["--model", model]
        let adapter = adapterPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !adapter.isEmpty { args += ["--adapter", adapter] }
        args += ["--output-dir", resolvedOutputDir]
        args += ["--prompt", prompt]
        args += ["--max-tokens", String(maxTokens)]
        args += ["--dpi", String(dpi)]
        if maxPages > 0 { args += ["--max-pages", String(maxPages)] }
        if repetitionPenalty != 1.0 {
            args += ["--repetition-penalty", String(repetitionPenalty)]
            args += ["--repetition-context-size", String(repetitionContextSize)]
        }
        return args
    }

    var hasInput: Bool {
        !inputPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
}
