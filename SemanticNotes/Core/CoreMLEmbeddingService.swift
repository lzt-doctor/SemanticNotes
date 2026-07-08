//
//  CoreMLEmbeddingService.swift
//  SemanticNotes
//

import CoreML
import Foundation
import Hub
import Tokenizers

/// multilingual-e5-small(Core ML / FP16)による埋め込みの実装。
/// なぜ actor か: 推論は数十 ms 単位のブロッキング処理なので、メインスレッドから
/// 隔離する。呼び出し側は async になり、UI を止めずに埋め込みを計算できる。
actor CoreMLEmbeddingService: EmbeddingService {
    enum EmbeddingError: Error {
        /// モデル or トークナイザが見つからない(scripts/install_model.sh 未実行など)
        case resourceNotFound(String)
        case invalidResource(String)
        case unexpectedOutput
    }

    /// モデル変換時の入力上限(convert_model.py の MAX_SEQ_LEN と一致させる)
    static let maxTokens = 512

    nonisolated let dimension = 384

    private let model: MLModel
    private let tokenizer: any Tokenizer

    init(bundle: Bundle = .main) throws {
        // Xcode が .mlpackage をコンパイルした .mlmodelc をバンドルから読む。
        // 自動生成クラスを使わないのは、モデル未配置の環境(CI)でも
        // コンパイルが通る状態を保つため。
        guard let modelURL = bundle.url(forResource: "MultilingualE5Small", withExtension: "mlmodelc") else {
            throw EmbeddingError.resourceNotFound("MultilingualE5Small.mlmodelc")
        }
        self.model = try MLModel(contentsOf: modelURL, configuration: MLModelConfiguration())
        self.tokenizer = try Self.loadTokenizer(from: bundle)
    }

    /// Python 側(validate_model.py)と同じ tokenizer.json を読み込む。
    /// なぜ同梱ファイルか: ネットワーク通信は制約で禁止。Hugging Face Hub からの
    /// 取得ではなく、変換時に書き出したファイルをアプリに同梱して読む。
    private static func loadTokenizer(from bundle: Bundle) throws -> any Tokenizer {
        guard let dataURL = bundle.url(forResource: "tokenizer", withExtension: "json"),
              let configURL = bundle.url(forResource: "tokenizer_config", withExtension: "json") else {
            throw EmbeddingError.resourceNotFound("tokenizer.json / tokenizer_config.json")
        }
        return try AutoTokenizer.from(
            tokenizerConfig: Config(try jsonDictionary(at: configURL)),
            tokenizerData: Config(try jsonDictionary(at: dataURL))
        )
    }

    private static func jsonDictionary(at url: URL) throws -> [NSString: Any] {
        guard let dictionary = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [NSString: Any] else {
            throw EmbeddingError.invalidResource(url.lastPathComponent)
        }
        return dictionary
    }

    /// トークン列を返す(Python 実装との一致検証テストでも使う)
    func tokenize(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    func embed(_ text: String) throws -> [Float] {
        var ids = tokenize(text)
        // 上限超過は黙った切り捨てで壊れるのではなく、</s> を保って明示的に切り詰める
        if ids.count > Self.maxTokens, let eos = ids.last {
            ids = Array(ids.prefix(Self.maxTokens - 1)) + [eos]
        }

        let shape = [1, NSNumber(value: ids.count)]
        let inputIDs = try MLMultiArray(shape: shape, dataType: .int32)
        let attentionMask = try MLMultiArray(shape: shape, dataType: .int32)
        for (index, id) in ids.enumerated() {
            inputIDs[index] = NSNumber(value: Int32(id))
            attentionMask[index] = 1 // バッチ=1でパディングなしなので全て実トークン
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
        ])
        let output = try model.prediction(from: input)
        guard let array = output.featureValue(for: "embedding")?.multiArrayValue else {
            throw EmbeddingError.unexpectedOutput
        }
        // 出力は変換時に FP32 指定・L2 正規化済み。そのまま [Float] に写す
        return array.withUnsafeBufferPointer(ofType: Float.self) { Array($0) }
    }
}
