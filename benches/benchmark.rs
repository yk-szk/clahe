use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use ndarray_rand::{rand_distr::Uniform, RandomExt};

fn exec_clahe(size: usize, tile_sample: f64, random_input: bool) -> Array2<u8> {
    let arr: Array2<u8> = if random_input {
        Array2::random((size, size), Uniform::new(0., 255.)).mapv(|e| e as u8)
    } else {
        Array2::zeros((size, size))
    };

    let grid_width = 8;
    let grid_height = 8;
    let clip_limit = 40;
    clahe::clahe_ndarray(arr.view(), grid_width, grid_height, clip_limit, tile_sample).unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("clahe size");
    for size in [256, 512, 768, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| exec_clahe(size, 1.0, true));
        });
    }
    group.finish();

    let mut group = c.benchmark_group("clahe sample");
    for tile_sample in 1..=8 {
        group.bench_with_input(
            BenchmarkId::from_parameter(tile_sample),
            &tile_sample,
            |b, &tile_sample| {
                b.iter(|| exec_clahe(1024, tile_sample as f64, true));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
