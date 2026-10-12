# FP-Growth Classification

This project uses FP-Growth to create simple classification rules. Instead of only printing frequent itemsets, it finds itemsets that point to a class label.

The data file should be comma-separated. The last value in each row is treated as the class label.

Example from the Iris data:

```text
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

The feature names are generic, so a row can become something like:

```text
attr1=low, attr2=high, attr3=low, attr4=low -> Iris-setosa
```

## Build

```sh
g++ -std=c++17 -Wall -Wextra -pedantic -O2 main.cpp -o fp_growth
```

## Run

```sh
./fp_growth <data.csv> <min_support> <min_confidence>
```

Example:

```sh
./fp_growth iris/iris.data 0.2 0.7
```

`min_support` controls how often a pattern has to appear in the dataset.

`min_confidence` controls how reliable a classification rule has to be.

## Output

The output prints classification rules:

```text
Transactions: 150

Classification rules
  {attr3=low, attr4=low} -> {Iris-setosa} support=0.333 confidence=1.000 count=50
```

This means the items on the left side are used to predict the class on the right side.
