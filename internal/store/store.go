// Package store implements the on-disk vector-corpus format used by vecrank.
//
// A corpus file is a line-oriented text format so that it stays diffable and
// inspectable:
//
//	# vecrank v1
//	dims <N>
//	count <M>
//	seed <S>
//	vec <id> <x1> <x2> ... <xN>
//
// Vector components are formatted with strconv.FormatFloat(f, 'f', 6, 64) so
// every value round-trips byte-exactly. IDs are arbitrary non-space tokens;
// the file order defines insertion order but never affects query output (all
// ranking is deterministic: score descending, then id ascending).
package store

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

// Vec is one labeled vector in the corpus.
type Vec struct {
	ID    string
	Comps []float64
}

// Corpus is the in-memory form of a corpus file.
type Corpus struct {
	Dims  int
	Seed  int64
	Vecs  []Vec
	index map[string]int
}

// Open reads a corpus file.
func Open(path string) (*Corpus, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	c := &Corpus{index: map[string]int{}}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		t := strings.TrimSpace(sc.Text())
		if t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		fields := strings.Fields(t)
		switch fields[0] {
		case "dims":
			if len(fields) != 2 {
				return nil, fmt.Errorf("line %d: dims takes one value", line)
			}
			c.Dims, err = strconv.Atoi(fields[1])
			if err != nil || c.Dims <= 0 {
				return nil, fmt.Errorf("line %d: bad dims", line)
			}
		case "count":
			// informational; recomputed on save
		case "seed":
			if len(fields) != 2 {
				return nil, fmt.Errorf("line %d: seed takes one value", line)
			}
			c.Seed, err = strconv.ParseInt(fields[1], 10, 64)
			if err != nil {
				return nil, fmt.Errorf("line %d: bad seed", line)
			}
		case "vec":
			if c.Dims == 0 {
				return nil, fmt.Errorf("line %d: vec before dims", line)
			}
			if len(fields) != 2+c.Dims {
				return nil, fmt.Errorf("line %d: expected %d components, got %d",
					line, c.Dims, len(fields)-2)
			}
			v := Vec{ID: fields[1]}
			for _, tok := range fields[2:] {
				x, err := strconv.ParseFloat(tok, 64)
				if err != nil {
					return nil, fmt.Errorf("line %d: bad component %q", line, tok)
				}
				v.Comps = append(v.Comps, x)
			}
			if _, dup := c.index[v.ID]; dup {
				return nil, fmt.Errorf("line %d: duplicate id %q", line, v.ID)
			}
			c.index[v.ID] = len(c.Vecs)
			c.Vecs = append(c.Vecs, v)
		default:
			return nil, fmt.Errorf("line %d: unknown key %q", line, fields[0])
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if c.Dims == 0 {
		return nil, fmt.Errorf("missing dims header")
	}
	return c, nil
}

// Save writes the corpus back in the canonical format.
func (c *Corpus) Save(path string) error {
	var b strings.Builder
	fmt.Fprintf(&b, "# vecrank v1\ndims %d\ncount %d\nseed %d\n",
		c.Dims, len(c.Vecs), c.Seed)
	for _, v := range c.Vecs {
		parts := make([]string, 0, len(v.Comps)+2)
		parts = append(parts, "vec", v.ID)
		for _, x := range v.Comps {
			parts = append(parts, FormatF(x))
		}
		b.WriteString(strings.Join(parts, " "))
		b.WriteByte('\n')
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// FormatF is the canonical float formatting used everywhere in vecrank.
func FormatF(x float64) string {
	return strconv.FormatFloat(x, 'f', 6, 64)
}

// Mulberry32 is a small, fully specified deterministic PRNG. Seeds map to the
// same sequence on every platform and every Go version; generation therefore
// depends only on (seed, count, dims).
type Mulberry32 struct{ state uint32 }

// NewMulberry32 seeds the generator.
func NewMulberry32(seed int64) *Mulberry32 {
	return &Mulberry32{state: uint32(seed)}
}

// Next returns the next uint32 in the sequence.
func (m *Mulberry32) Next() uint32 {
	m.state += 0x6D2B79F5
	var t = m.state
	t = (t << 15) | (t >> 17)
	t *= t | 1
	t ^= t + (t << 7) | (t >> 4)
	return t
}

// Float returns the next value in [-1, 1).
func (m *Mulberry32) Float() float64 {
	return float64(int32(m.Next())) / float64(1<<31)
}

// GenID returns deterministic id "v%05d" for ordinal i.
func GenID(i int) string { return fmt.Sprintf("v%05d", i) }

// Generate builds a corpus deterministically from (seed, count, dims).
// Component values live in [-1, 1) and are stored with 6-decimal precision,
// so the stored bytes are a pure function of the parameters.
func Generate(seed int64, count, dims int) *Corpus {
	rng := NewMulberry32(seed)
	c := &Corpus{Dims: dims, Seed: seed, index: map[string]int{}}
	for i := 0; i < count; i++ {
		comps := make([]float64, dims)
		for j := range comps {
			comps[j], _ = strconv.ParseFloat(FormatF(rng.Float()), 64)
		}
		id := GenID(i)
		c.index[id] = i
		c.Vecs = append(c.Vecs, Vec{ID: id, Comps: comps})
	}
	return c
}

// Metric is a similarity/distance measure. Higher Score = more similar for
// every metric, so ranking is uniformly "score descending".
type Metric string

const (
	Cosine    Metric = "cosine"
	Dot       Metric = "dot"
	Euclidean Metric = "euclidean"
)

// ValidMetric reports whether m is supported.
func ValidMetric(m Metric) bool {
	switch m {
	case Cosine, Dot, Euclidean:
		return true
	}
	return false
}

// Score computes the metric between a corpus vector and a query. Euclidean
// returns negated distance so that "higher is more similar" holds everywhere.
// For zero-norm inputs, cosine is defined as 0 (documented convention).
func Score(m Metric, v, q []float64) float64 {
	switch m {
	case Dot:
		s := 0.0
		for i := range v {
			s += v[i] * q[i]
		}
		return s
	case Cosine:
		var dot, nV, nQ float64
		for i := range v {
			dot += v[i] * q[i]
			nV += v[i] * v[i]
			nQ += q[i] * q[i]
		}
		if nV == 0 || nQ == 0 {
			return 0
		}
		return dot / (math.Sqrt(nV) * math.Sqrt(nQ))
	case Euclidean:
		s := 0.0
		for i := range v {
			d := v[i] - q[i]
			s += d * d
		}
		return -math.Sqrt(s)
	}
	panic("vecrank: unknown metric " + string(m))
}

// Hit is one ranked result.
type Hit struct {
	Rank  int
	ID    string
	Score float64
}

// Index returns the ordinal of id, or -1 when absent.
func (c *Corpus) Index(id string) (int, bool) {
	i, ok := c.index[id]
	return i, ok
}

// Append adds a labeled vector (caller validates uniqueness/dims).
func (c *Corpus) Append(id string, comps []float64) {
	c.index[id] = len(c.Vecs)
	c.Vecs = append(c.Vecs, Vec{ID: id, Comps: comps})
}

// Rank returns ALL corpus vectors ranked by the metric: score descending,
// ties broken by id ascending (byte order). rank starts at 1.
func (c *Corpus) Rank(m Metric, q []float64) []Hit {
	if len(q) != c.Dims {
		panic("vecrank: query dimension mismatch")
	}
	hits := make([]Hit, 0, len(c.Vecs))
	for _, v := range c.Vecs {
		hits = append(hits, Hit{ID: v.ID, Score: Score(m, v.Comps, q)})
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].ID < hits[j].ID
	})
	for i := range hits {
		hits[i].Rank = i + 1
	}
	return hits
}

// DimStat is one dimension's summary statistics.
type DimStat struct {
	Dim  int
	Min  float64
	Max  float64
	Mean float64
	Std  float64 // population standard deviation
}

// Stats summarizes every dimension. With zero vectors, every field is 0.
func (c *Corpus) Stats() []DimStat {
	out := make([]DimStat, c.Dims)
	n := float64(len(c.Vecs))
	for d := 0; d < c.Dims; d++ {
		var mn, mx, sum, sumsq float64
		first := true
		for _, v := range c.Vecs {
			x := v.Comps[d]
			if first || x < mn {
				mn = x
			}
			if first || x > mx {
				mx = x
			}
			first = false
			sum += x
			sumsq += x * x
		}
		if len(c.Vecs) == 0 {
			continue
		}
		mean := sum / n
		variance := sumsq/n - mean*mean
		if variance < 0 {
			variance = 0
		}
		out[d] = DimStat{Dim: d, Min: mn, Max: mx, Mean: mean, Std: math.Sqrt(variance)}
	}
	return out
}

// NormHistogram buckets L2 norms into the given number of equal-width bins
// spanning [0, maxNorm]. Counts are a pure function of the corpus.
func (c *Corpus) NormHistogram(bins int) (edges []float64, counts []int) {
	if bins <= 0 {
		panic("vecrank: bins must be positive")
	}
	var maxNorm float64
	norms := make([]float64, len(c.Vecs))
	for i, v := range c.Vecs {
		var s float64
		for _, x := range v.Comps {
			s += x * x
		}
		norms[i] = math.Sqrt(s)
		if norms[i] > maxNorm {
			maxNorm = norms[i]
		}
	}
	edges = make([]float64, bins+1)
	counts = make([]int, bins)
	for b := 0; b <= bins; b++ {
		edges[b] = float64(b) * maxNorm / float64(bins)
	}
	for _, nrm := range norms {
		b := int(nrm / (maxNorm / float64(bins)))
		if b >= bins {
			b = bins - 1
		}
		counts[b]++
	}
	return edges, counts
}

// Quantized is a scalar-quantized copy of a corpus: each component is mapped
// with the symmetric int8 scheme q = round((x - offset) / scale) clamped to
// [-127, 127], and Dequant reconstructs x' = offset + q*scale. Quantization
// parameters are a pure function of the corpus (per-dimension min/max), so
// the same corpus always quantizes to the same bytes.
type Quantized struct {
	Dims    int
	Scales  []float64
	Offsets []float64
	Codes   [][]int8
	Order   []string // ids in corpus order
}

// Quantize builds the int8 representation. scale is (max-min)/254 per dim
// (never zero: a constant dimension uses scale 1 to avoid division by zero).
func (c *Corpus) Quantize() *Quantized {
	q := &Quantized{Dims: c.Dims,
		Scales:  make([]float64, c.Dims),
		Offsets: make([]float64, c.Dims),
		Order:   make([]string, len(c.Vecs)),
		Codes:   make([][]int8, len(c.Vecs)),
	}
	stats := c.Stats()
	for d, s := range stats {
		span := s.Max - s.Min
		if span == 0 {
			span = 254 // constant dim: degenerate scale, codes all equal
		}
		q.Scales[d] = span / 254
		q.Offsets[d] = s.Min + (s.Max-s.Min)/2
	}
	for i, v := range c.Vecs {
		q.Order[i] = v.ID
		codes := make([]int8, c.Dims)
		for d, x := range v.Comps {
			f := math.Round((x - q.Offsets[d]) / q.Scales[d])
			if f > 127 {
				f = 127
			}
			if f < -127 {
				f = -127
			}
			codes[d] = int8(f)
		}
		q.Codes[i] = codes
	}
	return q
}

// Dequant reconstructs float components from a code row.
func (q *Quantized) Dequant(row int) []float64 {
	out := make([]float64, q.Dims)
	for d, code := range q.Codes[row] {
		out[d] = q.Offsets[d] + float64(code)*q.Scales[d]
	}
	return out
}

// ReconstructionError reports the max and mean absolute component error of
// dequantization across the whole corpus.
func (c *Corpus) ReconstructionError(q *Quantized) (maxAbs, meanAbs float64) {
	var sum float64
	for i, v := range c.Vecs {
		rec := q.Dequant(i)
		for d := range v.Comps {
			e := math.Abs(v.Comps[d] - rec[d])
			if e > maxAbs {
				maxAbs = e
			}
			sum += e
		}
	}
	if len(c.Vecs) > 0 && c.Dims > 0 {
		meanAbs = sum / float64(len(c.Vecs)*c.Dims)
	}
	return maxAbs, meanAbs
}

// QScore computes the metric between a quantized row and a float query using
// dequantized components — deterministic and consistent with Score.
func (q *Quantized) QScore(m Metric, row int, query []float64) float64 {
	return Score(m, q.Dequant(row), query)
}

// QRank ranks all quantized rows by the metric (same total order as Rank).
func (q *Quantized) QRank(m Metric, query []float64) []Hit {
	hits := make([]Hit, 0, len(q.Order))
	for i, id := range q.Order {
		hits = append(hits, Hit{ID: id, Score: q.QScore(m, i, query)})
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].ID < hits[j].ID
	})
	for i := range hits {
		hits[i].Rank = i + 1
	}
	return hits
}
