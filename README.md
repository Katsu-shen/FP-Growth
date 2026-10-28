# FP-Growth Classification

This project uses FP-Growth to create simple classification rules. Instead of only printing frequent itemsets, it finds itemsets that point to a class label.

The data file should be comma-separated with a header row. The final header names
the class column, and the final value in each data row is treated as the class label.

Example from the Iris data:

```text
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,Iris-setosa
```

The first four values are features, and `Iris-setosa` is the class.

## Numeric Values

FP-Growth works with item names, not raw decimal numbers. Because the Iris data uses numeric measurements, the program changes each numeric column into three equal-width bins:

```text
low
medium
high
```

The feature names come from the CSV header, so a row can become something like:

```text
sepal_length=low, sepal_width=high, petal_length=low, petal_width=low -> Iris-setosa
```

Missing values written as `?` are skipped, so they do not become frequent patterns.

## Build

The easiest way to build the project is:

```sh
make
```

The manual compile command is:

```sh
g++ -std=c++17 -Wall -Wextra -pedantic -O2 main.cpp data_loader.cpp preprocessing.cpp fp_tree.cpp classifier.cpp output.cpp -o fp_growth
```

## Optional CUDA build for Jetson Orin Nano

The normal `fp_growth` program is CPU-only. On a Jetson Orin Nano with the CUDA
toolkit installed, build the optional CUDA version with:

```sh
make cuda
```

This produces `fp_growth_cuda` and targets the Orin Nano GPU architecture (`sm_87`).
Set `CUDA_ARCH` only when building for a different NVIDIA GPU:

```sh
make cuda CUDA_ARCH=87
```

The CUDA executable mines frequent feature patterns and counts class rules on the GPU by
default. CSV loading, preprocessing, candidate generation, and output formatting remain on
the CPU. Use `--mining-backend cpu` to compare CPU FP-tree mining with CUDA mining, and use
`--backend cpu` to compare CPU class-rule counting. It fails clearly if CUDA is requested
but no usable NVIDIA CUDA device is available.

## Run

```sh
./fp_growth <data.csv> <min_support> <min_confidence>
```

Optional controls can focus the output and avoid generating very long patterns:

```sh
./fp_growth <data.csv> <min_support> <min_confidence> \
  --class <label> --max-length <count> --top <count>
```

Use `--no-header` for a legacy headerless CSV. Its feature names are generated
as `attr1`, `attr2`, and so on. `--header` is accepted explicitly but is the default.

The CUDA executable accepts independent mining and class-rule backends:

```sh
./fp_growth_cuda iris/iris.data 0.2 0.7 --mining-backend cuda --backend cuda
./fp_growth_cuda iris/iris.data 0.2 0.7 --mining-backend cpu --backend cpu
```

To verify the CUDA result manually on the Jetson, run the same command through
both executables and compare their output:

```sh
./fp_growth iris/iris.data 0.2 0.7 > iris_cpu.txt
./fp_growth_cuda iris/iris.data 0.2 0.7 --backend cuda > iris_cuda.txt
diff -u iris_cpu.txt iris_cuda.txt
```

Example:

```sh
./fp_growth iris/iris.data 0.2 0.7
```

Heart disease example:

```sh
./fp_growth heart+disease/processed.cleveland.data 0.2 0.7
```

Bank marketing example that only prints the top 50 `yes` rules with at most
three feature items:

```sh
./fp_growth bank+marketing/bank_clean.csv 0.01 0.6 --class yes --max-length 3 --top 50
```

You can also run the Iris example with:

```sh
make run-iris
```

## CPU Benchmark

The committed CPU benchmark records the prior string-item baseline against the
integer-ID FP-tree. Run the benchmark script from a branch under comparison:

```sh
python3 benchmarks/benchmark_cpu.py
```

The script creates a temporary `origin/main` worktree, builds `fp_growth` in both
worktrees, verifies byte-identical output, and writes CPU-only measurements to
`benchmarks/results_cpu.csv` and `benchmarks/report_cpu.md`.

## Next Optimization Steps

The CUDA miner is result-compatible with CPU FP-tree mining, but it still uses
candidate-by-transaction support checks. The highest-value next improvement is
to store transaction membership as vertical item bitsets and calculate support
with GPU bitwise intersections and population counts.

Before porting more preprocessing work to CUDA, profile CSV loading, feature
binning, candidate generation, GPU transfers, mining, and class-rule counting
separately. For wide one-hot datasets such as Covertype, convert related binary
columns into categorical items before mining; this reduces transaction width
without discarding feature identity.

For each optimization, compare CPU and CUDA output byte-for-byte and report at
least five warm-cache runs with median elapsed time, min–max range, peak RSS,
exact command options, and the Jetson/CUDA software versions.

`min_support` is the minimum fraction of all transactions that contain both
the feature pattern and its class label. For example, `0.02` requires at least
2% of all rows to match both sides of a rule.

`min_confidence` controls how reliable a classification rule has to be.

`--class` keeps rules for one class label. `--max-length` limits the number
of feature items on the left side of a rule. `--top` limits the printed rules
after they are sorted by confidence and support. `--mining-backend` selects
CPU FP-tree or CUDA frequent-pattern mining; `--backend` selects CPU or CUDA
class-rule counting when using `fp_growth_cuda`.

## Output

The output prints classification rules:

```text
Transactions: 150

Classification rules
  {petal_length=low, petal_width=low} -> {Iris-setosa} support=0.333 confidence=1.000 count=50
```

This means the items on the left side are used to predict the class on the right side.

## File Layout

The code is split into smaller files so each part has its own job:

```text
main.cpp            program flow
data_loader.cpp     reads and checks the csv file
preprocessing.cpp   bins numeric values into low, medium, and high
fp_tree.cpp         builds and mines the fp-tree
classifier.cpp      creates classification rules
output.cpp          handles printing and command-line parsing
```
