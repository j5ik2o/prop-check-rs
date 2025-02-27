pub mod choose;
pub mod one;

use crate::gen::choose::Choose;
use crate::gen::one::One;
use crate::rng::{NextRandValue, RNG};
use crate::state::State;
use bigdecimal::Num;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::rc::Rc;

/// Factory responsibility for generating Gens.
pub struct Gens;

impl Gens {
  /// Generates a Gen that returns `()`.
  ///
  /// # Returns
  /// * A `Gen<()>` that generates the unit value
  pub fn unit() -> Gen<()> {
    Self::pure(())
  }

  /// Generates a Gen that returns a constant value.
  ///
  /// # Arguments
  /// * `value` - The value to be wrapped in a Gen
  ///
  /// # Returns
  /// * A `Gen<B>` that always generates the provided value
  pub fn pure<B>(value: B) -> Gen<B>
  where
    B: Clone + 'static, {
    Gen::<B>::new(State::value(value))
  }

  /// Generates a Gen that returns a value from a lazily evaluated function.
  ///
  /// # Arguments
  /// * `f` - A closure that produces the value when called
  ///
  /// # Returns
  /// * A `Gen<B>` that generates the value by calling the provided function
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let expensive_computation = || {
  ///     // some expensive computation
  ///     42
  /// };
  /// let gen = Gens::pure_lazy(expensive_computation);
  /// ```
  pub fn pure_lazy<B, F>(f: F) -> Gen<B>
  where
    F: Fn() -> B + 'static,
    B: Clone + 'static, {
    Self::pure(()).map(move |_| f())
  }

  /// Generates a Gen that wraps the value of another Gen in Some.
  ///
  /// # Arguments
  /// * `gen` - The Gen whose values will be wrapped in Some
  ///
  /// # Returns
  /// * A `Gen<Option<B>>` that always generates Some containing the value from the input Gen
  ///
  /// # Type Parameters
  /// * `B` - The type of value to be wrapped in Some, must implement Clone and have a 'static lifetime
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let number_gen = Gens::choose(1, 10);
  /// let some_number_gen = Gens::some(number_gen);  // Generates Some(1)..Some(10)
  /// ```
  pub fn some<B>(gen: Gen<B>) -> Gen<Option<B>>
  where
    B: Clone + 'static, {
    gen.map(Some)
  }

  /// Generates a Gen that returns Some or None based on the value of Gen.
  /// The probability distribution is 90% for Some and 10% for None.
  ///
  /// # Arguments
  /// * `gen` - The Gen to be wrapped in an Option
  ///
  /// # Returns
  /// * A `Gen<Option<B>>` that generates Some(value) 90% of the time and None 10% of the time
  ///
  /// # Type Parameters
  /// * `B` - The type of value to be generated, must implement Debug, Clone and have a 'static lifetime
  pub fn option<B>(gen: Gen<B>) -> Gen<Option<B>>
  where
    B: Debug + Clone + 'static, {
    Self::frequency([(1, Self::pure(None)), (9, Self::some(gen))])
  }

  /// Generates a Gen that produces a Result type by combining two Gens.
  ///
  /// # Arguments
  /// * `gt` - The Gen that produces the Ok variant values
  /// * `ge` - The Gen that produces the Err variant values
  ///
  /// # Returns
  /// * A `Gen<Result<T, E>>` that randomly generates either Ok(T) or Err(E)
  ///
  /// # Type Parameters
  /// * `T` - The success type, must implement Choose, Clone and have a 'static lifetime
  /// * `E` - The error type, must implement Clone and have a 'static lifetime
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let success_gen = Gens::choose(1, 10);
  /// let error_gen = Gens::pure("error");
  /// let result_gen = Gens::either(success_gen, error_gen);
  /// ```
  pub fn either<T, E>(gt: Gen<T>, ge: Gen<E>) -> Gen<Result<T, E>>
  where
    T: Choose + Clone + 'static,
    E: Clone + 'static, {
    Self::one_of([gt.map(Ok), ge.map(Err)])
  }

  /// Generates a Gen that produces values according to specified weights.
  ///
  /// # Arguments
  /// * `values` - An iterator of tuples where the first element is the weight (u32) and
  ///             the second element is the value to be generated. The probability of each
  ///             value being generated is proportional to its weight.
  ///
  /// # Returns
  /// * A `Gen<B>` that generates values with probabilities determined by their weights
  ///
  /// # Panics
  /// * Panics if all weights are 0
  /// * Panics if no values are provided
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let weighted_gen = Gens::frequency_values([
  ///     (2, "common"),    // 2/3 probability
  ///     (1, "rare"),      // 1/3 probability
  /// ]);
  /// ```
  pub fn frequency_values<B>(values: impl IntoIterator<Item = (u32, B)>) -> Gen<B>
  where
    B: Debug + Clone + 'static, {
    Self::frequency(values.into_iter().map(|(n, value)| (n, Gens::pure(value))))
  }

  /// Generates a Gen that produces values from other Gens according to specified weights.
  ///
  /// # Arguments
  /// * `values` - An iterator of tuples where the first element is the weight (u32) and
  ///             the second element is another Gen. The probability of each Gen being
  ///             chosen is proportional to its weight.
  ///
  /// # Returns
  /// * A `Gen<B>` that generates values by selecting and running other Gens based on their weights
  ///
  /// # Panics
  /// * Panics if all weights are 0
  /// * Panics if no values are provided
  ///
  /// # Implementation Notes
  /// * Uses a BTreeMap for efficient weighted selection
  /// * Filters out entries with zero weight
  /// * Maintains cumulative weights for probability calculations
  pub fn frequency<B>(values: impl IntoIterator<Item = (u32, Gen<B>)>) -> Gen<B>
  where
    B: Debug + Clone + 'static, {
    // 事前に容量を確保するためにVecを使用
    let mut filtered = Vec::new();
    let mut total_weight = 0;

    // 重みが0より大きい項目だけをフィルタリング
    for (weight, value) in values.into_iter() {
      if weight > 0 {
        filtered.push((weight, value));
      }
    }

    // BTreeMapを構築（累積重みをキーとして使用）
    let mut tree = BTreeMap::new();
    for (weight, value) in filtered {
      total_weight += weight;
      tree.insert(total_weight, value);
    }

    // 空の場合はエラーを回避
    if total_weight == 0 {
      panic!("Empty frequency distribution");
    }

    // 乱数を生成して対応する値を返す
    Self::choose_u32(1, total_weight).flat_map(move |n| {
      // n以上の最小のキーを持つエントリを取得
      let entry = tree.range(n..).next().unwrap();
      entry.1.clone()
    })
  }

  /// Generates a Gen that produces a vector of n values generated by another Gen.
  ///
  /// # Arguments
  /// * `n` - The number of values to generate
  /// * `gen` - The Gen used to generate each value
  ///
  /// # Returns
  /// * A `Gen<Vec<B>>` that generates a vector containing n values
  ///
  /// # Performance
  /// * Uses lazy evaluation internally for better memory efficiency
  /// * For large n (>= 1000), consider using `list_of_n_chunked` or `list_of_n_chunked_optimal`
  ///   which may provide better performance through chunk-based processing
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let number_gen = Gens::choose(1, 100);
  /// let numbers_gen = Gens::list_of_n(5, number_gen);
  /// // Could generate vec![42, 17, 89, 3, 71]
  /// ```
  pub fn list_of_n<B>(n: usize, gen: Gen<B>) -> Gen<Vec<B>>
  where
    B: Clone + 'static, {
    Self::list_of_n_lazy(n, gen)
  }

  /// Generates a Gen whose elements are the values generated by the specified number of Gen,
  /// using the optimal chunk size based on benchmarks.
  pub fn list_of_n_chunked_optimal<B>(n: usize, gen: Gen<B>) -> Gen<Vec<B>>
  where
    B: Clone + 'static, {
    Self::list_of_n_chunked(n, usize::MAX, gen)
  }

  /// Generates a Gen that produces a vector of values using chunk-based processing for better performance.
  ///
  /// # Arguments
  /// * `n` - The number of values to generate
  /// * `chunk_size` - The size of chunks for batch processing. For optimal performance:
  ///                  - Use 1000 for large n (>= 1000)
  ///                  - For smaller n, the provided chunk_size is used as is
  /// * `gen` - The Gen used to generate each value
  ///
  /// # Returns
  /// * A `Gen<Vec<B>>` that generates a vector containing n values
  ///
  /// # Panics
  /// * Panics if chunk_size is 0
  ///
  /// # Performance Notes
  /// * Processes values in chunks to reduce memory allocation overhead
  /// * Automatically adjusts chunk size based on total number of elements
  /// * More efficient than `list_of_n` for large datasets
  /// * Consider using `list_of_n_chunked_optimal` for automatic chunk size optimization
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let number_gen = Gens::choose(1, 100);
  /// let numbers_gen = Gens::list_of_n_chunked(1000, 100, number_gen);
  /// ```
  pub fn list_of_n_chunked<B>(n: usize, chunk_size: usize, gen: Gen<B>) -> Gen<Vec<B>>
  where
    B: Clone + 'static, {
    if chunk_size == 0 {
      panic!("Chunk size must be greater than 0");
    }

    // ベンチマーク結果に基づいて最適なチャンクサイズを計算
    let optimal_chunk_size = if n < 1000 {
      // 小さいサイズの場合は指定されたチャンクサイズを使用
      chunk_size
    } else {
      // ベンチマーク結果から、1000が最も効率的なチャンクサイズ
      if chunk_size == usize::MAX {
        // デフォルト値（usize::MAX）が指定された場合は1000を使用
        1000
      } else {
        // 指定されたチャンクサイズを使用
        chunk_size
      }
    };

    Gen::<Vec<B>>::new(State::<RNG, Vec<B>>::new(move |rng: RNG| {
      let mut result = Vec::with_capacity(n);
      let mut current_rng = rng;

      // チャンク単位で処理
      for chunk_start in (0..n).step_by(optimal_chunk_size) {
        // 現在のチャンクのサイズを計算（最後のチャンクは小さくなる可能性がある）
        let current_chunk_size = std::cmp::min(optimal_chunk_size, n - chunk_start);

        // チャンクサイズ分のStateを生成
        let mut chunk_states: Vec<State<RNG, B>> = Vec::with_capacity(current_chunk_size);
        chunk_states.resize_with(current_chunk_size, || gen.clone().sample);

        // チャンクを処理
        let (chunk_result, new_rng) = State::sequence(chunk_states).run(current_rng);
        result.extend(chunk_result);
        current_rng = new_rng;
      }

      (result, current_rng)
    }))
  }

  /// Generates a Gen that produces a vector of values using lazy evaluation.
  ///
  /// # Arguments
  /// * `n` - The number of values to generate
  /// * `gen` - The Gen used to generate each value
  ///
  /// # Returns
  /// * A `Gen<Vec<B>>` that generates a vector containing n values
  ///
  /// # Performance Notes
  /// * Uses lazy evaluation to generate values one at a time
  /// * Minimizes memory usage by not pre-allocating all states
  /// * More memory efficient than eager evaluation for large n
  /// * May be slower than chunk-based processing for very large datasets
  /// * Maintains consistent memory usage regardless of n
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let number_gen = Gens::choose(1, 100);
  /// let numbers_gen = Gens::list_of_n_lazy(5, number_gen);
  /// ```
  pub fn list_of_n_lazy<B>(n: usize, gen: Gen<B>) -> Gen<Vec<B>>
  where
    B: Clone + 'static, {
    Gen::<Vec<B>>::new(State::<RNG, Vec<B>>::new(move |rng: RNG| {
      let mut result = Vec::with_capacity(n);
      let mut current_rng = rng;

      // 遅延評価を使用して値を生成
      for _ in 0..n {
        let (value, new_rng) = gen.clone().run(current_rng);
        result.push(value);
        current_rng = new_rng;
      }

      (result, current_rng)
    }))
  }

  /// Generates a Gen that returns a single value using the One trait.
  ///
  /// # Type Parameters
  /// * `T` - The type that implements the One trait
  ///
  /// # Returns
  /// * A `Gen<T>` that generates values using the One trait implementation
  ///
  /// # Implementation Notes
  /// * Uses the `one()` method from the One trait to generate values
  /// * Useful for types that have a natural "one" or "unit" value
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let number_gen = Gens::one::<i32>();  // Generates using i32's One implementation
  /// let bool_gen = Gens::one::<bool>();   // Generates using bool's One implementation
  /// ```
  pub fn one<T: One>() -> Gen<T> {
    One::one()
  }

  /// Generates a Gen that produces random i64 values.
  ///
  /// # Returns
  /// * A `Gen<i64>` that generates random 64-bit signed integers
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_i64();  // Generates random i64 values
  /// ```
  pub fn one_i64() -> Gen<i64> {
    Gen {
      sample: State::<RNG, i64>::new(move |rng: RNG| rng.next_i64()),
    }
  }

  /// Generates a Gen that produces random u64 values.
  ///
  /// # Returns
  /// * A `Gen<u64>` that generates random 64-bit unsigned integers
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_u64();  // Generates random u64 values
  /// ```
  pub fn one_u64() -> Gen<u64> {
    Gen {
      sample: State::<RNG, u64>::new(move |rng: RNG| rng.next_u64()),
    }
  }

  /// Generates a Gen that produces random i32 values.
  ///
  /// # Returns
  /// * A `Gen<i32>` that generates random 32-bit signed integers
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_i32();  // Generates random i32 values
  /// ```
  pub fn one_i32() -> Gen<i32> {
    Gen {
      sample: State::<RNG, i32>::new(move |rng: RNG| rng.next_i32()),
    }
  }

  /// Generates a Gen that produces random u32 values.
  ///
  /// # Returns
  /// * A `Gen<u32>` that generates random 32-bit unsigned integers
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_u32();  // Generates random u32 values
  /// ```
  pub fn one_u32() -> Gen<u32> {
    Gen {
      sample: State::<RNG, u32>::new(move |rng: RNG| rng.next_u32()),
    }
  }

  /// Generates a Gen that produces random i16 values.
  ///
  /// # Returns
  /// * A `Gen<i16>` that generates random 16-bit signed integers
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_i16();  // Generates random i16 values
  /// ```
  pub fn one_i16() -> Gen<i16> {
    Gen {
      sample: State::<RNG, i16>::new(move |rng: RNG| rng.next_i16()),
    }
  }

  /// Generates a Gen that produces random u16 values.
  ///
  /// # Returns
  /// * A `Gen<u16>` that generates random 16-bit unsigned integers
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_u16();  // Generates random u16 values
  /// ```
  pub fn one_u16() -> Gen<u16> {
    Gen {
      sample: State::<RNG, u16>::new(move |rng: RNG| rng.next_u16()),
    }
  }

  /// Generates a Gen that produces random i8 values.
  ///
  /// # Returns
  /// * A `Gen<i8>` that generates random 8-bit signed integers
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_i8();  // Generates random i8 values
  /// ```
  pub fn one_i8() -> Gen<i8> {
    Gen {
      sample: State::<RNG, i8>::new(move |rng: RNG| rng.next_i8()),
    }
  }

  /// Generates a Gen that produces random u8 values.
  ///
  /// # Returns
  /// * A `Gen<u8>` that generates random 8-bit unsigned integers
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_u8();  // Generates random u8 values
  /// ```
  pub fn one_u8() -> Gen<u8> {
    Gen {
      sample: State::<RNG, u8>::new(move |rng: RNG| rng.next_u8()),
    }
  }

  /// Generates a Gen that produces random char values.
  ///
  /// # Returns
  /// * A `Gen<char>` that generates random characters
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_char();  // Generates random characters
  /// ```
  pub fn one_char() -> Gen<char> {
    Self::one_u8().map(|v| v as char)
  }

  /// Generates a Gen that produces random boolean values.
  ///
  /// # Returns
  /// * A `Gen<bool>` that generates random true/false values
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_bool();  // Generates random booleans
  /// ```
  pub fn one_bool() -> Gen<bool> {
    Gen {
      sample: State::<RNG, bool>::new(|rng: RNG| rng.next_bool()),
    }
  }

  /// Generates a Gen that produces random f64 values.
  ///
  /// # Returns
  /// * A `Gen<f64>` that generates random 64-bit floating point numbers
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let gen = Gens::one_f64();  // Generates random f64 values
  /// ```
  pub fn one_f64() -> Gen<f64> {
    Gen {
      sample: State::<RNG, f64>::new(move |rng: RNG| rng.next_f64()),
    }
  }
/// Generates a Gen that produces random f32 values.
///
/// # Returns
/// * A `Gen<f32>` that generates random 32-bit floating point numbers
///
/// # Examples
/// ```
/// use prop_check_rs::gen::Gens;
/// let gen = Gens::one_f32();  // Generates random f32 values
/// ```
pub fn one_f32() -> Gen<f32> {
  Gen {
    sample: State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()),
  }
}

  /// Generates a Gen that produces values by randomly selecting from other Gens.
  ///
  /// # Arguments
  /// * `values` - An iterator of Gens to choose from, with equal probability
  ///
  /// # Returns
  /// * A `Gen<T>` that generates values by randomly selecting and running one of the input Gens
  ///
  /// # Type Parameters
  /// * `T` - The type of value to generate, must implement Choose, Clone and have a 'static lifetime
  ///
  /// # Panics
  /// * Panics if the input iterator is empty
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let small = Gens::choose(1, 10);
  /// let large = Gens::choose(100, 200);
  /// let combined = Gens::one_of([small, large]);  // 50% chance of each range
  /// ```
  pub fn one_of<T: Choose + Clone + 'static>(values: impl IntoIterator<Item = Gen<T>>) -> Gen<T> {
    // 直接Vecに変換
    let vec: Vec<_> = values.into_iter().collect();

    // 空の場合はエラーを回避
    if vec.is_empty() {
      panic!("Empty one_of distribution");
    }

    // インデックスを選択して対応する値を返す
    Self::choose(0usize, vec.len() - 1).flat_map(move |idx| {
      let gen = &vec[idx as usize];
      gen.clone()
    })
  }

  /// Generates a Gen that randomly selects from a set of fixed values.
  ///
  /// # Arguments
  /// * `values` - An iterator of values to choose from, with equal probability
  ///
  /// # Returns
  /// * A `Gen<T>` that generates values by randomly selecting one from the input set
  ///
  /// # Type Parameters
  /// * `T` - The type of value to generate, must implement Choose, Clone and have a 'static lifetime
  ///
  /// # Panics
  /// * Panics if the input iterator is empty
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// // Note: &str doesn't implement Choose trait
  /// // let colors = Gens::one_of_values(["red", "green", "blue"]);
  /// let numbers = Gens::one_of_values([1, 10, 100]);  // Equal probability for each number
  /// ```
  pub fn one_of_values<T: Choose + Clone + 'static>(values: impl IntoIterator<Item = T>) -> Gen<T> {
    // 直接Vecに変換
    let vec: Vec<_> = values.into_iter().collect();

    // 空の場合はエラーを回避
    if vec.is_empty() {
      panic!("Empty one_of_values distribution");
    }

    // インデックスを選択して対応する値を返す
    Self::choose(0usize, vec.len() - 1).map(move |idx| vec[idx as usize].clone())
  }

  /// Generates a Gen that returns one randomly selected value from the specified maximum and minimum ranges of generic type.
  pub fn choose<T: Choose>(min: T, max: T) -> Gen<T> {
    Choose::choose(min, max)
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type char.
  pub fn choose_char(min: char, max: char) -> Gen<char> {
    // 文字コードを使用して範囲を計算
    let min_code = min as u32;
    let max_code = max as u32;

    // 範囲が無効な場合はエラーを回避
    if min_code > max_code {
      panic!("Invalid char range");
    }

    // 文字コードの範囲から選択して文字に変換
    Self::choose_u32(min_code, max_code).map(move |code| std::char::from_u32(code).unwrap_or(min))
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type i64.
  ///
  /// # Arguments
  /// * `min` - The minimum value (inclusive) of the range
  /// * `max` - The maximum value (inclusive) of the range
  ///
  /// # Returns
  /// * A `Gen<i64>` that generates random i64 values in the range [min, max]
  ///
  /// # Panics
  /// * Panics if `min > max` (invalid range)
  pub fn choose_i64(min: i64, max: i64) -> Gen<i64> {
    if min > max {
      panic!("Invalid range: min > max");
    }
    
    // 範囲の大きさを計算
    let range = max - min + 1;
    
    // オーバーフローを防ぐために絶対値を使用
    Gen {
      sample: State::<RNG, i64>::new(move |rng: RNG| {
        let (n, new_rng) = rng.next_i64();
        // 負の値を正の値に変換し、範囲内に収める
        let abs_n = if n < 0 { -n } else { n };
        let value = min + (abs_n % range);
        (value, new_rng)
      }),
    }
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type u64.
  pub fn choose_u64(min: u64, max: u64) -> Gen<u64> {
    Gen {
      sample: State::<RNG, u64>::new(move |rng: RNG| rng.next_u64()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type i32.
  ///
  /// # Arguments
  /// * `min` - The minimum value (inclusive) of the range
  /// * `max` - The maximum value (inclusive) of the range
  ///
  /// # Returns
  /// * A `Gen<i32>` that generates random i32 values in the range [min, max]
  ///
  /// # Panics
  /// * Panics if `min > max` (invalid range)
  pub fn choose_i32(min: i32, max: i32) -> Gen<i32> {
    if min > max {
      panic!("Invalid range: min > max");
    }
    
    // 範囲の大きさを計算
    let range = max - min + 1;
    
    // オーバーフローを防ぐために絶対値を使用
    Gen {
      sample: State::<RNG, i32>::new(move |rng: RNG| {
        let (n, new_rng) = rng.next_i32();
        // 負の値を正の値に変換し、範囲内に収める
        let abs_n = if n < 0 { -n } else { n };
        let value = min + (abs_n % range);
        (value, new_rng)
      }),
    }
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type u32.
  pub fn choose_u32(min: u32, max: u32) -> Gen<u32> {
    Gen {
      sample: State::<RNG, u32>::new(move |rng: RNG| rng.next_u32()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type i16.
  pub fn choose_i16(min: i16, max: i16) -> Gen<i16> {
    Gen {
      sample: State::<RNG, i16>::new(move |rng: RNG| rng.next_i16()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type u16.
  pub fn choose_u16(min: u16, max: u16) -> Gen<u16> {
    Gen {
      sample: State::<RNG, u16>::new(move |rng: RNG| rng.next_u16()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type i8.
  pub fn choose_i8(min: i8, max: i8) -> Gen<i8> {
    Gen {
      sample: State::<RNG, i8>::new(move |rng: RNG| rng.next_i8()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type u8.
  pub fn choose_u8(min: u8, max: u8) -> Gen<u8> {
    Gen {
      sample: State::<RNG, u8>::new(move |rng: RNG| rng.next_u8()),
    }
    .map(move |n| min + n % (max - min + 1))
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type f64.
  ///
  /// # Arguments
  /// * `min` - The minimum value (inclusive) of the range
  /// * `max` - The maximum value (inclusive) of the range
  ///
  /// # Returns
  /// * A `Gen<f64>` that generates random f64 values in the range [min, max]
  ///
  /// # Note
  /// * The distribution is uniform across the range
  pub fn choose_f64(min: f64, max: f64) -> Gen<f64> {
    Gen {
      sample: State::<RNG, f64>::new(move |rng: RNG| rng.next_f64()),
    }
    .map(move |d| min + d * (max - min))
  }

  /// Generates a Gen that returns one randomly selected value from a specified maximum and minimum range of type f32.
  ///
  /// # Arguments
  /// * `min` - The minimum value (inclusive) of the range
  /// * `max` - The maximum value (inclusive) of the range
  ///
  /// # Returns
  /// * A `Gen<f32>` that generates random f32 values in the range [min, max]
  ///
  /// # Note
  /// * The distribution is uniform across the range
  pub fn choose_f32(min: f32, max: f32) -> Gen<f32> {
    Gen {
      sample: State::<RNG, f32>::new(move |rng: RNG| rng.next_f32()),
    }
    .map(move |d| min + d * (max - min))
  }

  /// Generates a Gen that returns randomly selected even numbers from a specified range.
  ///
  /// # Arguments
  /// * `start` - The inclusive start of the range
  /// * `stop_exclusive` - The exclusive end of the range
  ///
  /// # Type Parameters
  /// * `T` - A numeric type that implements Choose, Num, Copy, and 'static
  ///
  /// # Returns
  /// * A `Gen<T>` that generates even numbers in the range [start, stop_exclusive)
  ///
  /// # Implementation Notes
  /// * If the start value is odd, the first even number after it will be used
  /// * If stop_exclusive is odd, stop_exclusive - 1 will be used as the end of the range
  /// * Maintains uniform distribution over even numbers in the range
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let even_gen = Gens::even(1, 10);  // Generates from {2, 4, 6, 8}
  /// ```
  pub fn even<T: Choose + Num + Copy + 'static>(start: T, stop_exclusive: T) -> Gen<T> {
    let two = T::one().add(T::one());
    Self::choose(
      start,
      if stop_exclusive % two == T::zero() {
        stop_exclusive - T::one()
      } else {
        stop_exclusive
      },
    )
    .map(move |n| if n % two != T::zero() { n + T::one() } else { n })
  }

  /// Generates a Gen that returns randomly selected odd numbers from a specified range.
  ///
  /// # Arguments
  /// * `start` - The inclusive start of the range
  /// * `stop_exclusive` - The exclusive end of the range
  ///
  /// # Type Parameters
  /// * `T` - A numeric type that implements Choose, Num, Copy, and 'static
  ///
  /// # Returns
  /// * A `Gen<T>` that generates odd numbers in the range [start, stop_exclusive)
  ///
  /// # Implementation Notes
  /// * If the start value is even, the first odd number after it will be used
  /// * If stop_exclusive is even, stop_exclusive - 1 will be used as the end of the range
  /// * Maintains uniform distribution over odd numbers in the range
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let odd_gen = Gens::odd(1, 10);  // Generates from {1, 3, 5, 7, 9}
  /// ```
  pub fn odd<T: Choose + Num + Copy + 'static>(start: T, stop_exclusive: T) -> Gen<T> {
    let two = T::one().add(T::one());
    Self::choose(
      start,
      if stop_exclusive % two != T::zero() {
        stop_exclusive - T::one()
      } else {
        stop_exclusive
      },
    )
    .map(move |n| if n % two == T::zero() { n + T::one() } else { n })
  }
}

/// Generator that generates values.
#[derive(Debug)]
pub struct Gen<A> {
  sample: State<RNG, A>,
}

impl<A: Clone + 'static> Clone for Gen<A> {
  fn clone(&self) -> Self {
    Self {
      sample: self.sample.clone(),
    }
  }
}

impl<A: Clone + 'static> Gen<A> {
  /// Evaluates the Gen with a given RNG to produce a value.
  ///
  /// # Arguments
  /// * `rng` - The random number generator to use
  ///
  /// # Returns
  /// * A tuple `(A, RNG)` containing the generated value and the updated RNG state
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// use prop_check_rs::rng::RNG;
  /// let gen = Gens::choose(1, 10);
  /// let (value, new_rng) = gen.run(RNG::new());
  /// assert!(value >= 1 && value <= 10);
  /// ```
  pub fn run(self, rng: RNG) -> (A, RNG) {
    self.sample.run(rng)
  }

  /// Creates a new Gen by wrapping a State.
  ///
  /// # Arguments
  /// * `b` - The State to wrap in a Gen
  ///
  /// # Returns
  /// * A new `Gen<B>` containing the provided State
  pub fn new<B>(b: State<RNG, B>) -> Gen<B> {
    Gen { sample: b }
  }

  /// Transforms the output of a Gen using a function.
  ///
  /// # Arguments
  /// * `f` - A function to apply to the generated value
  ///
  /// # Returns
  /// * A new `Gen<B>` that applies the function to generated values
  ///
  /// # Type Parameters
  /// * `B` - The result type after applying the function
  /// * `F` - The function type, must be Fn(A) -> B + 'static
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let numbers = Gens::choose(1, 10);
  /// let doubled = numbers.map(|x| x * 2);  // Generates numbers from 2 to 20
  /// ```
  pub fn map<B, F>(self, f: F) -> Gen<B>
  where
    F: Fn(A) -> B + 'static,
    B: Clone + 'static, {
    Self::new(self.sample.map(f))
  }

  /// Combines two Gens using a function that takes both of their results.
  ///
  /// # Arguments
  /// * `g` - Another Gen to combine with this one
  /// * `f` - A function that combines the results of both Gens
  ///
  /// # Returns
  /// * A new `Gen<C>` that combines the results of both Gens
  ///
  /// # Type Parameters
  /// * `B` - The type of the second Gen's output
  /// * `C` - The result type after combining both outputs
  /// * `F` - The function type, must be Fn(A, B) -> C + 'static
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let x = Gens::choose(1, 5);
  /// let y = Gens::choose(6, 10);
  /// let sum = x.and_then(y, |a, b| a + b);  // Generates sums from 7 to 15
  /// ```
  pub fn and_then<B, C, F>(self, g: Gen<B>, f: F) -> Gen<C>
  where
    F: Fn(A, B) -> C + 'static,
    A: Clone,
    B: Clone + 'static,
    C: Clone + 'static, {
    Self::new(self.sample.and_then(g.sample).map(move |(a, b)| f(a, b)))
  }

  /// Chains this Gen with a function that returns another Gen.
  ///
  /// # Arguments
  /// * `f` - A function that takes the result of this Gen and returns a new Gen
  ///
  /// # Returns
  /// * A new `Gen<B>` that represents the chained computation
  ///
  /// # Type Parameters
  /// * `B` - The type of the resulting Gen
  /// * `F` - The function type, must be Fn(A) -> Gen<B> + 'static
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::Gens;
  /// let numbers = Gens::choose(1, 3);
  /// let repeated = numbers.flat_map(|n| Gens::list_of_n(n, Gens::pure(n)));
  /// ```
  pub fn flat_map<B, F>(self, f: F) -> Gen<B>
  where
    F: Fn(A) -> Gen<B> + 'static,
    B: Clone + 'static, {
    Self::new(self.sample.flat_map(move |a| f(a).sample))
  }
}

/// Generator with size information.
pub enum SGen<A> {
  /// Generator with size information.
  Sized(Rc<RefCell<dyn Fn(u32) -> Gen<A>>>),
  /// Generator without size information.
  Unsized(Gen<A>),
}

impl<A: Clone + 'static> Clone for SGen<A> {
  fn clone(&self) -> Self {
    match self {
      SGen::Sized(f) => SGen::Sized(f.clone()),
      SGen::Unsized(g) => SGen::Unsized(g.clone()),
    }
  }
}

impl<A: Clone + 'static> SGen<A> {
  /// Creates a sized generator that can produce different Gens based on a size parameter.
  ///
  /// # Arguments
  /// * `f` - A function that takes a size parameter and returns a Gen
  ///
  /// # Returns
  /// * An `SGen<A>` that can generate size-dependent values
  ///
  /// # Type Parameters
  /// * `F` - The function type, must be Fn(u32) -> Gen<A> + 'static
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::SGen;
  /// use prop_check_rs::gen::Gens;
  /// let sized_gen = SGen::of_sized(|size| Gens::list_of_n(size as usize, Gens::choose(1, 10)));
  /// ```
  pub fn of_sized<F>(f: F) -> SGen<A>
  where
    F: Fn(u32) -> Gen<A> + 'static, {
    SGen::Sized(Rc::new(RefCell::new(f)))
  }

  /// Creates an unsized generator that wraps a fixed Gen.
  ///
  /// # Arguments
  /// * `gen` - The Gen to wrap
  ///
  /// # Returns
  /// * An `SGen<A>` that always uses the provided Gen
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::SGen;
  /// use prop_check_rs::gen::Gens;
  /// let fixed_gen = SGen::of_unsized(Gens::choose(1, 10));
  /// ```
  pub fn of_unsized(gen: Gen<A>) -> SGen<A> {
    SGen::Unsized(gen)
  }

  /// Runs the generator with an optional size parameter to produce a Gen.
  ///
  /// # Arguments
  /// * `i` - Optional size parameter, required for Sized variants
  ///
  /// # Returns
  /// * A `Gen<A>` that can generate values
  ///
  /// # Panics
  /// * Panics if a Sized variant is run without a size parameter
  ///
  /// # Examples
  /// ```
  /// use prop_check_rs::gen::SGen;
  /// use prop_check_rs::gen::Gens;
  /// let sized_gen = SGen::of_sized(|n| Gens::list_of_n(n as usize, Gens::pure(1)));
  /// let gen = sized_gen.run(Some(5));  // Creates a Gen that produces a vector of 5 ones
  /// ```
  pub fn run(&self, i: Option<u32>) -> Gen<A> {
    match self {
      SGen::Sized(f) => {
        let mf = f.borrow_mut();
        mf(i.unwrap())
      }
      SGen::Unsized(g) => g.clone(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::prop;
  use anyhow::Result;

  use std::cell::RefCell;
  use std::collections::HashMap;
  use std::env;
  use std::rc::Rc;

  fn init() {
    env::set_var("RUST_LOG", "info");
    let _ = env_logger::builder().is_test(true).try_init();
  }

  fn new_rng() -> RNG {
    RNG::new()
  }

  pub mod laws {
    use super::*;

    #[test]
    fn test_left_identity_law() -> Result<()> {
      init();
      let gen = Gens::choose_i32(1, i32::MAX / 2).map(|e| (RNG::new().with_seed(e as u64), e));
      let f = |x| Gens::pure(x);
      let laws_prop = prop::for_all_gen(gen, move |(s, n)| {
        Gens::pure(n).flat_map(f).run(s.clone()) == f(n).run(s)
      });
      prop::test_with_prop(laws_prop, 1, 100, new_rng())
    }

    #[test]
    fn test_right_identity_law() -> Result<()> {
      init();
      let gen = Gens::choose_i32(1, i32::MAX / 2).map(|e| (RNG::new().with_seed(e as u64), e));

      let laws_prop = prop::for_all_gen(gen, move |(s, x)| {
        Gens::pure(x).flat_map(|y| Gens::pure(y)).run(s.clone()) == Gens::pure(x).run(s)
      });

      prop::test_with_prop(laws_prop, 1, 100, new_rng())
    }

    #[test]
    fn test_associativity_law() -> Result<()> {
      init();
      let gen = Gens::choose_i32(1, i32::MAX / 2).map(|e| (RNG::new().with_seed(e as u64), e));
      let f = |x| Gens::pure(x * 2);
      let g = |x| Gens::pure(x + 1);
      let laws_prop = prop::for_all_gen(gen, move |(s, x)| {
        Gens::pure(x).flat_map(f).flat_map(g).run(s.clone()) == f(x).flat_map(g).run(s)
      });
      prop::test_with_prop(laws_prop, 1, 100, new_rng())
    }
  }

  #[test]
  fn test_frequency() -> Result<()> {
    let result = Rc::new(RefCell::new(HashMap::new()));
    let cloned_map = result.clone();

    let gens = [
      (1, Gens::choose(1u32, 10)),
      (3, Gens::choose(50u32, 100)),
      (1, Gens::choose(200u32, 300)),
    ];
    let gen = Gens::frequency(gens);
    let prop = prop::for_all_gen(gen, move |a| {
      let mut map = result.borrow_mut();
      log::info!("a: {}", a);
      if a >= 1 && a <= 10 {
        let r = map.entry(1).or_insert_with(|| 0);
        *r += 1;
        true
      } else if a >= 50 && a <= 100 {
        let r = map.entry(2).or_insert_with(|| 0);
        *r += 1;
        true
      } else if a >= 200 && a <= 300 {
        let r = map.entry(3).or_insert_with(|| 0);
        *r += 1;
        true
      } else {
        false
      }
    });
    let r = prop::test_with_prop(prop, 1, 100, new_rng());

    let map = cloned_map.borrow();
    let a_count = map.get(&1).unwrap();
    let b_count = map.get(&2).unwrap();
    let c_count = map.get(&3).unwrap();

    assert_eq!(*a_count + *b_count + *c_count, 100);
    println!("{cloned_map:?}");
    r
  }

  #[test]
  fn test_frequency_values() -> Result<()> {
    let result = Rc::new(RefCell::new(HashMap::new()));
    let cloned_map = result.clone();

    let gens = [(1, "a"), (1, "b"), (8, "c")];
    let gen = Gens::frequency_values(gens);
    let prop = prop::for_all_gen(gen, move |a| {
      let mut map = result.borrow_mut();
      let r = map.entry(a).or_insert_with(|| 0);
      *r += 1;
      true
    });
    let r = prop::test_with_prop(prop, 1, 100, new_rng());

    let map = cloned_map.borrow();
    let a_count = map.get(&"a").unwrap();
    let b_count = map.get(&"b").unwrap();
    let c_count = map.get(&"c").unwrap();

    assert_eq!(*a_count + *b_count + *c_count, 100);
    println!("{cloned_map:?}");
    r
  }

  #[test]
  fn test_list_of_n_chunked() {
    init();
    // 小さなチャンクサイズでテスト
    let n = 100;
    let chunk_size = 10;
    let gen = Gens::one_i32();
    let chunked_gen = Gens::list_of_n_chunked(n, chunk_size, gen.clone());

    let (result, _) = chunked_gen.run(new_rng());

    // 結果のサイズが正しいことを確認
    assert_eq!(result.len(), n);

    // 大きなチャンクサイズでテスト
    let large_chunk_size = 1000;
    let large_chunked_gen = Gens::list_of_n_chunked(n, large_chunk_size, gen);

    let (large_result, _) = large_chunked_gen.run(new_rng());

    // 結果のサイズが正しいことを確認
    assert_eq!(large_result.len(), n);
  }

  #[test]
  fn test_list_of_n_lazy() {
    init();
    let n = 100;
    let gen = Gens::one_i32();
    let lazy_gen = Gens::list_of_n_lazy(n, gen.clone());

    let (result, _) = lazy_gen.run(new_rng());

    // 結果のサイズが正しいことを確認
    assert_eq!(result.len(), n);

    // 通常のlist_of_nと結果を比較
    let normal_gen = Gens::list_of_n(n, gen);
    let (normal_result, _) = normal_gen.run(new_rng().with_seed(42));
    let (lazy_result, _) = Gens::list_of_n_lazy(n, Gens::one_i32()).run(new_rng().with_seed(42));

    // 同じシードを使用した場合、結果が同じであることを確認
    assert_eq!(normal_result, lazy_result);
  }

  #[test]
  fn test_large_data_generation() {
    init();
    // 大量のデータを生成
    let n = 10000;

    // 通常の方法
    let start_time = std::time::Instant::now();
    let gen = Gens::one_i32();
    let normal_gen = Gens::list_of_n(n, gen.clone());
    let (normal_result, _) = normal_gen.run(new_rng());
    let normal_duration = start_time.elapsed();

    // チャンク処理を使用
    let start_time = std::time::Instant::now();
    let chunk_size = 1000;
    let chunked_gen = Gens::list_of_n_chunked(n, chunk_size, gen.clone());
    let (chunked_result, _) = chunked_gen.run(new_rng());
    let chunked_duration = start_time.elapsed();

    // 遅延評価を使用
    let start_time = std::time::Instant::now();
    let lazy_gen = Gens::list_of_n_lazy(n, gen);
    let (lazy_result, _) = lazy_gen.run(new_rng());
    let lazy_duration = start_time.elapsed();

    // 結果のサイズが正しいことを確認
    assert_eq!(normal_result.len(), n);
    assert_eq!(chunked_result.len(), n);
    assert_eq!(lazy_result.len(), n);

    // パフォーマンス情報をログに出力
    log::info!("Normal generation time: {:?}", normal_duration);
    log::info!("Chunked generation time: {:?}", chunked_duration);
    log::info!("Lazy generation time: {:?}", lazy_duration);
  }
}
