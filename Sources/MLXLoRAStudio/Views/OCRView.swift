import AppKit
import SwiftUI

/// OCR page: run Unlimited-OCR over a PDF/image file or a folder (recursive),
/// optionally with a trained LoRA adapter, and watch progress live. Output is
/// one Markdown file per input under the chosen (or run) folder.
struct OCRView: View {
    @Bindable var store: AppStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                HeaderView(
                    title: "OCR",
                    subtitle: "Document parsing with Unlimited-OCR",
                    symbol: "doc.text.viewfinder"
                )

                // Single Start/Cancel pill, mirroring the Synthetic page: the
                // toolbar pair is reserved for the training runner.
                let isRunning = store.ocrRunner.isRunning
                Button {
                    if isRunning {
                        store.ocrRunner.stop()
                    } else {
                        Task { await store.startOCR() }
                    }
                } label: {
                    Label(
                        isRunning ? "Cancel OCR" : "Run OCR",
                        systemImage: isRunning ? "xmark.circle.fill" : "doc.text.viewfinder"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(isRunning ? .red : .accentColor)
                .controlSize(.large)
                .disabled(!isRunning && !store.ocr.hasInput)

                OCRRunConsolePill(runner: store.ocrRunner)

                OCRInputSection(config: $store.ocr)
                OCRModelSection(config: $store.ocr)
                OCROptionsSection(config: $store.ocr)
                OCROutputSection(
                    config: $store.ocr,
                    lastRunFolder: store.ocrRunner.lastRunFolder
                )
            }
            .padding(24)
        }
        .frame(minWidth: 360, idealWidth: 620, maxWidth: .infinity)
        .liquidGlass(cornerRadius: 18)
        .padding(16)
        .navigationTitle("OCR")
    }
}

// MARK: - Sections

private struct OCRInputSection: View {
    @Binding var config: OCRConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Input")
            TextField("PDF/image file or folder", text: $config.inputPath)
            HStack(spacing: 10) {
                Button("Choose File…") { pick(directories: false) }
                Button("Choose Folder…") { pick(directories: true) }
            }
            ToggleRow("Recurse into subfolders", isOn: $config.recursive)
            Text("Point at a single PDF/image, or a folder of them. PDFs are OCR'd page by page.")
                .foregroundStyle(.secondary)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
        }
        .formBlock()
    }

    private func pick(directories: Bool) {
        if let path = OCRFilePicker.choose(directories: directories) {
            config.inputPath = path
        }
    }
}

private struct OCRModelSection: View {
    @Binding var config: OCRConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Model")
            HFAssetPicker(
                text: $config.model,
                kind: .model,
                placeholder: "OCR model or HF repo (e.g. baidu/Unlimited-OCR)",
                footer: "Needs the mlx-vlm fork with the unlimited_ocr package installed."
            )
            HStack(spacing: 10) {
                TextField("Trained LoRA adapter (optional)", text: $config.adapterPath)
                Button("Choose…") {
                    if let path = OCRFilePicker.choose(directories: false) {
                        config.adapterPath = path
                    }
                }
            }
        }
        .formBlock()
    }
}

private struct OCROptionsSection: View {
    @Binding var config: OCRConfig

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Options")
            TextField("Prompt", text: $config.prompt)
            HStack {
                NumberField("Max tokens", value: $config.maxTokens)
                NumberField("PDF DPI", value: $config.dpi)
                NumberField("Max pages (0 = all)", value: $config.maxPages)
            }
            HStack {
                FloatingField("Repetition penalty", value: $config.repetitionPenalty)
                NumberField("Repetition context", value: $config.repetitionContextSize)
            }
            Text("Set repetition penalty above 1.0 (e.g. 1.05) to curb runaway tables on dense pages.")
                .foregroundStyle(.secondary)
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
        }
        .formBlock()
    }
}

private struct OCROutputSection: View {
    @Binding var config: OCRConfig
    let lastRunFolder: String

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            SectionTitle("Output")
            HStack(spacing: 10) {
                TextField("Output folder (optional)", text: $config.outputDir)
                Button("Choose…") {
                    if let path = OCRFilePicker.choose(directories: true) {
                        config.outputDir = path
                    }
                }
            }
            Text("Leave empty to collect Markdown outputs in a fresh run folder.")
                .foregroundStyle(.secondary)
                .font(.callout)
            if !lastRunFolder.isEmpty {
                InfoPill(text: "Open output", symbol: "folder", openPath: lastRunFolder)
            }
        }
        .formBlock()
    }
}

// MARK: - Console

private struct OCRRunConsolePill: View {
    @Bindable var runner: PythonJobRunner

    private var recentLines: [String] { Array(runner.logLines.suffix(6)) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Circle()
                    .fill(runner.isRunning ? .green : .secondary)
                    .frame(width: 8, height: 8)
                Text(runner.isRunning ? "Live Run" : "Run Console")
                    .font(.headline)
                Text(runner.currentCommand.isEmpty ? "Ready" : runner.currentCommand)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer()
                Button {
                    runner.clearTerminal()
                } label: {
                    Label("Clear", systemImage: "eraser").labelStyle(.iconOnly)
                }
                .buttonStyle(.borderless)
                .disabled(runner.logLines.isEmpty)
            }

            RunProgressBar(runner: runner)

            if recentLines.isEmpty {
                Text("Ready")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(recentLines.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(line.contains("[Studio]") ? .secondary : .primary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(.quaternary.opacity(0.45), lineWidth: 1)
        )
        .animation(.easeInOut(duration: 0.2), value: runner.isRunning)
        .animation(.easeInOut(duration: 0.18), value: runner.logLines.count)
    }
}

// MARK: - File picker

enum OCRFilePicker {
    /// Present an open panel for a file or a directory; returns the chosen path.
    @MainActor static func choose(directories: Bool) -> String? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = !directories
        panel.canChooseDirectories = directories
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        guard panel.runModal() == .OK, let url = panel.url else { return nil }
        return url.path
    }
}
