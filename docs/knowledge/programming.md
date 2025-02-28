# プログラミングルール

- コーディング規約は言語の標準に従う（詳細なフォーマットなどはフォーマッターで自動的適用される前提）
- デバッグは、ログ出力を適切にいれて、ステップバイステップで解析する
- コードを編集したら `cargo test` を実行してテストがパスすることを確認する
- モジュールは2018方式を採用してください
- オブジェクト間の循環参照は避けてください
- rustdocは英語で記述する
  - rustdocがないものは新規に追加する。既存のrustdocでも以下に該当しないものは是正すること
  - コードを見れば分かることは書かない。Why/Why notを中心に記載すること
  - タイトル, 型引数, 引数, 戻り値, パニック, その他注意事項など
  - 書き方は以下を参考にする。項目の順番は統一すること。
  ```rust
  /// Execute the Prop.
  ///
  /// # Arguments
  /// - `max_size` - The maximum size of the generated value.
  /// - `test_cases` - The number of test cases.
  /// - `rng` - The random number generator.
  ///
  /// # Returns
  /// - `PropResult` - The result of the Prop.
  ```
