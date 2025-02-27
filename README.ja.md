# prop-check-rs

prop-check-rsは、Rustで書かれたプロパティベーステスト用のライブラリです。関数型プログラミングの概念を活用し、テストデータの生成と検証を効率的に行うことができます。

*他の言語で読む: [English](README.md)*

[![Workflow Status](https://github.com/j5ik2o/prop-check-rs/workflows/ci/badge.svg)](https://github.com/j5ik2o/prop-check-rs/actions?query=workflow%3A%22ci%22)
[![crates.io](https://img.shields.io/crates/v/prop-check-rs.svg)](https://crates.io/crates/prop-check-rs)
[![docs.rs](https://docs.rs/prop-check-rs/badge.svg)](https://docs.rs/prop-check-rs)
[![tokei](https://tokei.rs/b1/github/j5ik2o/prop-check-rs)](https://github.com/XAMPPRocky/tokei)

## プロパティベーステストとは

プロパティベーステストは、特定の入力値に対するテストではなく、プログラムが満たすべき性質（プロパティ）を定義し、ランダムに生成された多数の入力値に対してそのプロパティが成り立つかを検証するテスト手法です。これにより、開発者が想定していないエッジケースを発見しやすくなります。

## 特徴

- 豊富なジェネレータ：様々な型のテストデータを簡単に生成
- 関数型プログラミングスタイル：モナドを活用した合成可能なAPI
- 状態ベースのテスト：状態機械のシミュレーションをサポート
- 高度なカスタマイズ：独自のジェネレータやプロパティを定義可能

## インストール

Cargo.tomlに以下を追加してください：

```toml
[dependencies]
prop-check-rs = "0.0.862"
```

## 基本的な使い方

### 1. 単純なプロパティテスト

以下は、リストの長さに関するプロパティをテストする例です：

```rust
use prop_check_rs::gen::Gens;
use prop_check_rs::prop::{for_all_gen, test_with_prop};
use prop_check_rs::rng::RNG;
use anyhow::Result;

#[test]
fn test_list_length_property() -> Result<()> {
    // 0から100までの整数のリストを生成するジェネレータ
    let gen = Gens::list_of_n(10, Gens::choose_i32(0, 100));
    
    // プロパティ：リストの長さは常に10
    let prop = for_all_gen(gen, |list| {
        list.len() == 10
    });
    
    // プロパティをテスト（最大サイズ1、100回のテストケース）
    test_with_prop(prop, 1, 100, RNG::new())
}
```

### 2. 値の選択

```rust
use prop_check_rs::gen::Gens;
use prop_check_rs::prop::{for_all_gen, test_with_prop};
use prop_check_rs::rng::RNG;
use anyhow::Result;

#[test]
fn test_one_of() -> Result<()> {
    // 指定された文字のいずれかを選択するジェネレータ
    let gen = Gens::one_of_values(['a', 'b', 'c', 'x', 'y', 'z']);
    
    // プロパティ：選択された文字は常に指定された文字のいずれか
    let prop = for_all_gen(gen, move |value| {
        log::info!("value = {}", value);
        ['a', 'b', 'c', 'x', 'y', 'z'].contains(&value)
    });
    
    test_with_prop(prop, 1, 100, RNG::new())
}
```

### 3. サイズ付きジェネレータの使用

```rust
use prop_check_rs::gen::Gens;
use prop_check_rs::prop::{for_all_gen_for_size, test_with_prop};
use prop_check_rs::rng::RNG;
use anyhow::Result;

#[test]
fn test_sized_generator() -> Result<()> {
    let gen = Gens::one_of_values(['a', 'b', 'c', 'x', 'y', 'z']);
    
    // サイズに基づいてリストを生成
    let prop = for_all_gen_for_size(
        move |size| Gens::list_of_n(size as usize, gen.clone()),
        move || {
            move |list| {
                // リストの長さがサイズと一致することを確認
                log::info!("list = {:?}", list);
                true
            }
        },
    );
    
    // 最大サイズ10、100回のテストケース
    test_with_prop(prop, 10, 100, RNG::new())
}
```

## 主要なコンポーネント

### Gen<A>

`Gen<A>`は型`A`の値を生成するジェネレータです。`map`、`flat_map`、`and_then`などのメソッドを提供し、既存のジェネレータから新しいジェネレータを作成できます。

```rust
// 1から100までの整数を生成するジェネレータ
let int_gen = Gens::choose_i32(1, 100);

// 整数を文字列に変換するジェネレータ
let string_gen = int_gen.map(|n| n.to_string());
```

### Gens

`Gens`は様々なジェネレータを作成するためのファクトリです。以下のようなジェネレータを提供しています：

- 基本型（整数、浮動小数点、文字、真偽値など）
- リスト
- オプション値
- 複数の選択肢から一つを選ぶ
- 確率に基づく選択

```rust
// 基本型のジェネレータ
let int_gen = Gens::one_i32();
let float_gen = Gens::one_f64();
let bool_gen = Gens::one_bool();

// 範囲を指定したジェネレータ
let range_gen = Gens::choose_i32(1, 100);

// リストのジェネレータ
let list_gen = Gens::list_of_n(10, range_gen);

// 複数の選択肢から一つを選ぶジェネレータ
let choice_gen = Gens::one_of_values(["apple", "banana", "orange"]);

// 確率に基づくジェネレータ
let weighted_gen = Gens::frequency_values([(1, "rare"), (5, "common"), (2, "uncommon")]);
```

### Prop

`Prop`はプロパティを表す構造体です。プロパティは、ジェネレータによって生成された値に対して検証する条件を定義します。

```rust
// 整数が常に正であることを検証するプロパティ
let positive_prop = for_all_gen(Gens::choose_i32(1, 100), |n| n > 0);

// 複数のプロパティを組み合わせる
let combined_prop = positive_prop.and(another_prop);
```

### State<S, A>

`State<S, A>`は状態`S`を持ち、値`A`を生成する計算を表すモナドです。これにより、状態を保持しながら計算を合成することができます。

```rust
// 状態を取得する
let get_state = State::<i32, i32>::get();

// 状態を設定する
let set_state = State::<i32, ()>::set(42);

// 状態を変更する
let modify_state = State::<i32, ()>::modify(|s| s + 1);

// 状態付き計算を合成する
let computation = get_state.flat_map(|s| {
    if s > 0 {
        State::pure(s * 2)
    } else {
        State::pure(0)
    }
});
```

## 高度な使用例

### 状態機械のテスト

`machine.rs`モジュールは、状態機械をシミュレートする例を提供しています。以下は、キャンディ販売機のシミュレーション例です：

```rust
// 入力
enum Input {
    Coin,
    Turn,
}

// 状態機械
struct Machine {
    locked: bool,
    candies: i32,
    coins: i32,
}

// 状態機械のシミュレーション
let inputs = vec![Input::Coin, Input::Turn, Input::Coin, Input::Turn];
let simulation = Machine::simulate_machine(inputs);
let result = simulation.run(Machine { locked: true, candies: 5, coins: 10 });
```

### カスタムジェネレータの作成

独自のジェネレータを作成することで、特定のドメインに特化したテストデータを生成できます：

```rust
// 有効なメールアドレスを生成するジェネレータ
fn email_gen() -> Gen<String> {
    let username_gen = Gens::list_of_n(8, Gens::choose_char('a', 'z'))
        .map(|chars| chars.into_iter().collect::<String>());
    
    let domain_gen = Gens::one_of_values(["example.com", "test.org", "mail.net"]);
    
    username_gen.and_then(domain_gen, |username, domain| {
        format!("{}@{}", username, domain)
    })
}

// 使用例
let prop = for_all_gen(email_gen(), |email| {
    // メールアドレスの検証ロジック
    email.contains('@')
});
```

## ベンチマーク

prop-check-rsは、大量のテストデータを効率的に生成するための最適化が施されています。ベンチマークを実行するには：

```bash
cargo bench
```

## ライセンス

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.