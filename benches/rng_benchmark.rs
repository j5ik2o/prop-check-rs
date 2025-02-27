use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use prop_check_rs::rng::RNG;

// 現在の実装
fn i32s_current(count: u32) -> Vec<i32> {
    let rng = RNG::new();
    let (result, _) = rng.i32s(count);
    result
}

// 最適化案1: Rc<RefCell<>>を使わずに直接StdRngを使用する実装
fn i32s_optimized1(count: u32) -> Vec<i32> {
    use rand::prelude::*;
    use rand::SeedableRng;
    
    let mut rng = StdRng::seed_from_u64(0);
    let mut result = Vec::with_capacity(count as usize);
    
    for _ in 0..count {
        result.push(rng.random::<i32>());
    }
    
    result
}

// 最適化案2: 並列処理を使用する実装
fn i32s_optimized2(count: u32) -> Vec<i32> {
    use rand::prelude::*;
    use rand::SeedableRng;
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let num_threads = num_cpus::get().min(8); // スレッド数を制限
    let chunk_size = count / num_threads as u32;
    let remainder = count % num_threads as u32;
    
    let result = Arc::new(Mutex::new(Vec::with_capacity(count as usize)));
    
    let mut handles = vec![];
    
    for i in 0..num_threads {
        let result_clone = Arc::clone(&result);
        let mut thread_count = chunk_size;
        
        // 最後のスレッドに余りを追加
        if i == num_threads - 1 {
            thread_count += remainder;
        }
        
        let handle = thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(i as u64); // 各スレッドで異なるシード
            let mut local_result = Vec::with_capacity(thread_count as usize);
            
            for _ in 0..thread_count {
                local_result.push(rng.random::<i32>());
            }
            
            let mut result = result_clone.lock().unwrap();
            result.extend(local_result);
        });
        
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    Arc::try_unwrap(result).unwrap().into_inner().unwrap()
}

// 最適化案3: イテレータを使用する実装
fn i32s_optimized3(count: u32) -> Vec<i32> {
    use rand::prelude::*;
    use rand::SeedableRng;
    
    let mut rng = StdRng::seed_from_u64(0);
    (0..count).map(|_| rng.random::<i32>()).collect()
}

fn bench_i32s(c: &mut Criterion) {
    let mut group = c.benchmark_group("i32s");
    
    // 小さいサイズでのベンチマーク
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("current", size), size, |b, &size| {
            b.iter(|| i32s_current(black_box(size)))
        });
        
        group.bench_with_input(BenchmarkId::new("optimized1", size), size, |b, &size| {
            b.iter(|| i32s_optimized1(black_box(size)))
        });
        
        group.bench_with_input(BenchmarkId::new("optimized2", size), size, |b, &size| {
            b.iter(|| i32s_optimized2(black_box(size)))
        });
        
        group.bench_with_input(BenchmarkId::new("optimized3", size), size, |b, &size| {
            b.iter(|| i32s_optimized3(black_box(size)))
        });
    }
    
    // 大きいサイズでのベンチマーク（時間がかかるので注意）
    for size in [100000, 1000000].iter() {
        group.bench_with_input(BenchmarkId::new("current", size), size, |b, &size| {
            b.iter(|| i32s_current(black_box(size)))
        });
        
        group.bench_with_input(BenchmarkId::new("optimized1", size), size, |b, &size| {
            b.iter(|| i32s_optimized1(black_box(size)))
        });
        
        group.bench_with_input(BenchmarkId::new("optimized2", size), size, |b, &size| {
            b.iter(|| i32s_optimized2(black_box(size)))
        });
        
        group.bench_with_input(BenchmarkId::new("optimized3", size), size, |b, &size| {
            b.iter(|| i32s_optimized3(black_box(size)))
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_i32s);
criterion_main!(benches);