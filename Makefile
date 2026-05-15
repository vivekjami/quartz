.PHONY: dev build test bench fmt lint clean

dev:
	maturin develop --release

build:
	maturin build --release

test:
	cargo test --workspace
	pytest tests/ -v

bench:
	python -m quartz.eval.profiler --index data/index/ --n-queries 1000
	python -m benchmarks.wand_ablation --index data/index/ --n-queries 500
	python -m benchmarks.ef_search_pareto --hnsw-dir data/hnsw/ --n-queries 200

fmt:
	cargo fmt --all
	ruff format quartz/ benchmarks/ tests/

lint:
	cargo clippy --workspace -- -D warnings
	ruff check quartz/ benchmarks/ tests/
	mypy quartz/

clean:
	cargo clean
	rm -rf data/index/ data/hnsw/ target/ results/
