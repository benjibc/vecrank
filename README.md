# vecrank

`vecrank` is a small, fully deterministic vector-similarity CLI. It maintains a
frozen corpus of labeled float vectors and answers exact similarity queries:
full ranked lists, top-k lists, or a single k-th order statistic.

Everything is byte-stable: vector components and scores are printed with
exactly 6 decimals, ranking is score descending with ties broken by id
ascending (byte order), corpus generation is a pure function of
`(seed, count, dims)`, and no command reads the wall clock, randomness, or the
environment. The same inputs always produce the same bytes on stdout.

## Commands

```sh
# generate a frozen corpus deterministically
vecrank new corpus.txt --seed 42 --count 200 --dims 8

# append labeled vectors from a text file ("id x1 x2 ..." per line)
vecrank add corpus.txt --file extra.txt

# full ranked list (cosine similarity), or top-k, as table/csv/json
vecrank query corpus.txt --q 0.5,-0.25,1,0,0.3,-0.7,0.1,0.9 --metric cosine
vecrank query corpus.txt --q ... --metric dot --k 5 --format csv
vecrank query corpus.txt --q ... --metric euclidean --format json

# the k-th ranked entry only (order statistic; k=1 best, k=count worst)
vecrank nth corpus.txt --q ... --metric cosine --k 12

# answer every query in a file ("label f1,f2,..." per line), grouped output
vecrank batch corpus.txt --file queries.txt --metric cosine --k 3 --format csv

# per-dimension statistics + L2-norm histogram
vecrank stat corpus.txt --bins 10 --format json

# symmetric int8 scalar quantization: per-dim scale/offset, reconstruction error
vecrank quantize corpus.txt --error

# rank the QUANTIZED corpus (dequantized scoring) — the int8 approximation path
vecrank qrank corpus.txt --q ... --metric cosine --k 3

# corpus metadata
vecrank info corpus.txt
```

## Conventions

- Metrics: `cosine` (of a zero-norm vector is defined as 0), `dot`, and
  `euclidean` (printed as negated distance, so every metric is
  "higher = more similar").
- Ranking is total and deterministic: score descending, then id ascending.
- Quantization is symmetric int8 per dimension: `q = round((x - offset) /
  scale)` clamped to [-127, 127], with `scale = (max-min)/254` and
  `offset = (max+min)/2` computed per dimension; a constant dimension uses the
  degenerate scale 1. The same corpus always quantizes to the same codes.
- The corpus file format is line-oriented text (`dims`/`count`/`seed` headers
  plus one `vec <id> <components...>` line per vector), diffable and
  inspectable; `add` rewrites it in canonical form.

## Build

```sh
go build -o vecrank ./cmd/vecrank
go test ./...
```

## License

MIT.
