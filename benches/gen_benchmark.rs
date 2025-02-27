use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prop_check_rs::gen::Gens;
use prop_check_rs::rng::RNG;

// list_of_nメソッドのベンチマーク
fn bench_list_of_n(c: &mut Criterion) {
  let mut group = c.benchmark_group("list_of_n");

  // 小さいサイズでのベンチマーク
  for size in [10, 100, 1000].iter() {
    group.bench_with_input(BenchmarkId::new("list_of_n", size), size, |b, &size| {
      b.iter(|| {
        let gen = Gens::one_i32();
        let list_gen = Gens::list_of_n(size, gen);
        let (_, _) = list_gen.run(RNG::new());
      })
    });

    group.bench_with_input(BenchmarkId::new("list_of_n_chunked", size), size, |b, &size| {
      b.iter(|| {
        let gen = Gens::one_i32();
        let chunk_size = (size / 10).max(1);
        let list_gen = Gens::list_of_n_chunked(size, chunk_size, gen);
        let (_, _) = list_gen.run(RNG::new());
      })
    });

    group.bench_with_input(BenchmarkId::new("list_of_n_lazy", size), size, |b, &size| {
      b.iter(|| {
        let gen = Gens::one_i32();
        let list_gen = Gens::list_of_n_lazy(size, gen);
        let (_, _) = list_gen.run(RNG::new());
      })
    });
  }

  // 大きいサイズでのベンチマーク
  for size in [10000, 50000].iter() {
    group.bench_with_input(BenchmarkId::new("list_of_n", size), size, |b, &size| {
      b.iter(|| {
        let gen = Gens::one_i32();
        let list_gen = Gens::list_of_n(size, gen);
        let (_, _) = list_gen.run(RNG::new());
      })
    });

    group.bench_with_input(BenchmarkId::new("list_of_n_chunked", size), size, |b, &size| {
      b.iter(|| {
        let gen = Gens::one_i32();
        let chunk_size = (size / 10).max(1);
        let list_gen = Gens::list_of_n_chunked(size, chunk_size, gen);
        let (_, _) = list_gen.run(RNG::new());
      })
    });

    group.bench_with_input(BenchmarkId::new("list_of_n_lazy", size), size, |b, &size| {
      b.iter(|| {
        let gen = Gens::one_i32();
        let list_gen = Gens::list_of_n_lazy(size, gen);
        let (_, _) = list_gen.run(RNG::new());
      })
    });
  }

  group.finish();
}

// 異なるチャンクサイズでのlist_of_n_chunkedのベンチマーク
fn bench_chunk_sizes(c: &mut Criterion) {
  let mut group = c.benchmark_group("chunk_sizes");
  let size = 10000;

  for chunk_size in [1, 10, 100, 1000, 5000, 10000].iter() {
    group.bench_with_input(
      BenchmarkId::new("chunk_size", chunk_size),
      chunk_size,
      |b, &chunk_size| {
        b.iter(|| {
          let gen = Gens::one_i32();
          let list_gen = Gens::list_of_n_chunked(size, chunk_size, gen);
          let (_, _) = list_gen.run(RNG::new());
        })
      },
    );
  }

  group.finish();
}

// 異なる型のジェネレータのベンチマーク
fn bench_different_types(c: &mut Criterion) {
  let mut group = c.benchmark_group("different_types");
  let size = 1000;

  group.bench_function("i32", |b| {
    b.iter(|| {
      let gen = Gens::one_i32();
      let list_gen = Gens::list_of_n(size, gen);
      let (_, _) = list_gen.run(RNG::new());
    })
  });

  group.bench_function("f64", |b| {
    b.iter(|| {
      let gen = Gens::one_f64();
      let list_gen = Gens::list_of_n(size, gen);
      let (_, _) = list_gen.run(RNG::new());
    })
  });

  group.bench_function("bool", |b| {
    b.iter(|| {
      let gen = Gens::one_bool();
      let list_gen = Gens::list_of_n(size, gen);
      let (_, _) = list_gen.run(RNG::new());
    })
  });

  group.bench_function("char", |b| {
    b.iter(|| {
      let gen = Gens::one_char();
      let list_gen = Gens::list_of_n(size, gen);
      let (_, _) = list_gen.run(RNG::new());
    })
  });

  group.finish();
}

criterion_group!(benches, bench_list_of_n, bench_chunk_sizes, bench_different_types);
criterion_main!(benches);
