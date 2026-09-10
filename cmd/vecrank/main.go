// vecrank is a small, fully deterministic vector-similarity CLI.
//
// It maintains a frozen corpus of labeled float vectors and answers exact
// similarity queries against it: full ranked lists, top-k lists, or a single
// k-th order statistic. All output formatting is byte-stable: floats are
// printed with 6 decimals, ties break by id ascending, and no command depends
// on wall-clock time, randomness, or environment variables.
package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/benjibc/vecrank/internal/store"
)

const usage = `vecrank - deterministic vector-similarity ranking over a frozen corpus

Usage:
  vecrank new <corpus> --seed S --count N --dims D
      Generate a frozen corpus deterministically from (seed, count, dims).
      Vector ids are v00000..v<N-1>; components live in [-1, 1).

  vecrank add <corpus> --file <vectors.txt>
      Append labeled vectors from a text file: one "id x1 x2 ..." per line.
      Lines starting with # are comments. Fails on duplicate ids or wrong
      dimensionality. Rewrites the corpus in canonical form.

  vecrank query <corpus> --q f1,f2,... --metric cosine|dot|euclidean [--k K] [--format table|csv|json]
      Rank the whole corpus against the query (score descending, ties by id
      ascending). --k limits to the top K (default: all). --format selects the
      output layout (default table).

  vecrank nth <corpus> --q f1,f2,... --metric cosine|dot|euclidean --k K
      Print exactly the K-th ranked entry (1-based). This is the order
      statistic: k=1 is the best match, k=count is the worst.

  vecrank batch <corpus> --file <queries.txt> --metric cosine|dot|euclidean
                    [--k K] [--format table|csv|json]
      Answer every query in the file (one "label f1,f2,..." per line; #
      comments). Output is grouped by query in file order, each group preceded
      by "query <label>" in table/csv mode or nested under the label in json
      mode. --k applies per query.

  vecrank stat <corpus> [--bins B] [--format table|csv|json]
      Per-dimension summary statistics (min/max/mean/std) plus an L2-norm
      histogram over B equal-width bins (default 10).

  vecrank quantize <corpus> [--format table|csv|json] [--error]
      Report the symmetric int8 scalar quantization of the corpus: per-dim
      scale and offset, and (with --error) the max and mean absolute
      reconstruction error over all components.

  vecrank qrank <corpus> --q f1,f2,... --metric cosine|dot|euclidean [--k K]
      Rank the QUANTIZED corpus (dequantized scoring) — the deterministic
      int8 approximation path. Same ordering conventions as query.

  vecrank info <corpus>
      Print dims, count, and seed of the corpus.

Conventions:
  - cosine of a zero-norm vector is defined as 0.
  - euclidean prints negated distance so every metric is "higher = closer".
  - floats are printed with exactly 6 decimals everywhere.
`

func main() {
	if len(os.Args) < 2 {
		fail(usage)
	}
	var err error
	switch os.Args[1] {
	case "new":
		err = cmdNew(os.Args[2:])
	case "add":
		err = cmdAdd(os.Args[2:])
	case "query":
		err = cmdQuery(os.Args[2:])
	case "nth":
		err = cmdNth(os.Args[2:])
	case "batch":
		err = cmdBatch(os.Args[2:])
	case "stat":
		err = cmdStat(os.Args[2:])
	case "quantize":
		err = cmdQuantize(os.Args[2:])
	case "qrank":
		err = cmdQRank(os.Args[2:])
	case "info":
		err = cmdInfo(os.Args[2:])
	case "help", "--help", "-h":
		fmt.Print(usage)
		return
	default:
		fail("unknown command %q\n\n%s", os.Args[1], usage)
	}
	if err != nil {
		fail("vecrank: %v", err)
	}
}

func fail(f string, a ...any) {
	fmt.Fprintf(os.Stderr, f+"\n", a...)
	os.Exit(1)
}

// flagval extracts --name value from an args tail, with a default.
func flagval(args []string, name, def string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "--"+name {
			return args[i+1]
		}
	}
	return def
}

func positional(args []string) (pos []string) {
	for i := 0; i < len(args); i++ {
		if strings.HasPrefix(args[i], "--") {
			i++ // skip its value
			continue
		}
		pos = append(pos, args[i])
	}
	return pos
}

func cmdNew(args []string) error {
	pos := positional(args)
	if len(pos) != 1 {
		return fmt.Errorf("new takes exactly one corpus path")
	}
	seed, err := strconv.ParseInt(flagval(args, "seed", "0"), 10, 64)
	if err != nil {
		return fmt.Errorf("bad --seed: %v", err)
	}
	count, err := strconv.Atoi(flagval(args, "count", "0"))
	if err != nil || count <= 0 {
		return fmt.Errorf("--count must be a positive integer")
	}
	dims, err := strconv.Atoi(flagval(args, "dims", "0"))
	if err != nil || dims <= 0 {
		return fmt.Errorf("--dims must be a positive integer")
	}
	c := store.Generate(seed, count, dims)
	if err := c.Save(pos[0]); err != nil {
		return err
	}
	fmt.Printf("wrote %s (dims=%d count=%d seed=%d)\n", pos[0], c.Dims, len(c.Vecs), c.Seed)
	return nil
}

func cmdAdd(args []string) error {
	pos := positional(args)
	if len(pos) != 1 {
		return fmt.Errorf("add takes exactly one corpus path")
	}
	c, err := store.Open(pos[0])
	if err != nil {
		return err
	}
	file := flagval(args, "file", "")
	if file == "" {
		return fmt.Errorf("add requires --file")
	}
	data, err := os.ReadFile(file)
	if err != nil {
		return err
	}
	added := 0
	for ln, line := range strings.Split(string(data), "\n") {
		t := strings.TrimSpace(line)
		if t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		fields := strings.Fields(t)
		if len(fields) != 2+c.Dims {
			return fmt.Errorf("%s line %d: expected %d components, got %d",
				file, ln+1, c.Dims, len(fields)-2)
		}
		if _, dup := c.Index(fields[0]); dup {
			return fmt.Errorf("%s line %d: duplicate id %q", file, ln+1, fields[0])
		}
		comps := make([]float64, c.Dims)
		for i, tok := range fields[2:] {
			comps[i], err = strconv.ParseFloat(tok, 64)
			if err != nil {
				return fmt.Errorf("%s line %d: bad component %q", file, ln+1, tok)
			}
		}
		c.Append(fields[0], comps)
		added++
	}
	if err := c.Save(pos[0]); err != nil {
		return err
	}
	fmt.Printf("added %d vector(s); count=%d\n", added, len(c.Vecs))
	return nil
}

func parseQuery(s string, dims int) ([]float64, error) {
	parts := strings.Split(s, ",")
	if len(parts) != dims {
		return nil, fmt.Errorf("query has %d components, corpus has %d", len(parts), dims)
	}
	q := make([]float64, dims)
	for i, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, fmt.Errorf("bad query component %q", p)
		}
		q[i] = v
	}
	return q, nil
}

func printHits(format string, hits []store.Hit) {
	switch format {
	case "csv":
		fmt.Println("rank,id,score")
		for _, h := range hits {
			fmt.Printf("%d,%s,%s\n", h.Rank, h.ID, store.FormatF(h.Score))
		}
	case "json":
		var b strings.Builder
		b.WriteString("[")
		for i, h := range hits {
			if i > 0 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, `{"rank":%d,"id":%q,"score":%s}`,
				h.Rank, h.ID, store.FormatF(h.Score))
		}
		b.WriteString("]\n")
		fmt.Print(b.String())
	default: // table
		fmt.Println("rank\tscore\tid")
		for _, h := range hits {
			fmt.Printf("%d\t%s\t%s\n", h.Rank, store.FormatF(h.Score), h.ID)
		}
	}
}

func cmdQuery(args []string) error {
	pos := positional(args)
	if len(pos) != 1 {
		return fmt.Errorf("query takes exactly one corpus path")
	}
	c, err := store.Open(pos[0])
	if err != nil {
		return err
	}
	q, err := parseQuery(flagval(args, "q", ""), c.Dims)
	if err != nil {
		return err
	}
	m := store.Metric(flagval(args, "metric", "cosine"))
	if !store.ValidMetric(m) {
		return fmt.Errorf("unknown metric %q (cosine|dot|euclidean)", m)
	}
	hits := c.Rank(m, q)
	if k := flagval(args, "k", "0"); k != "0" {
		n, err := strconv.Atoi(k)
		if err != nil || n <= 0 || n > len(hits) {
			return fmt.Errorf("--k must be between 1 and %d", len(hits))
		}
		hits = hits[:n]
	}
	printHits(flagval(args, "format", "table"), hits)
	return nil
}

func cmdNth(args []string) error {
	pos := positional(args)
	if len(pos) != 1 {
		return fmt.Errorf("nth takes exactly one corpus path")
	}
	c, err := store.Open(pos[0])
	if err != nil {
		return err
	}
	q, err := parseQuery(flagval(args, "q", ""), c.Dims)
	if err != nil {
		return err
	}
	m := store.Metric(flagval(args, "metric", "cosine"))
	if !store.ValidMetric(m) {
		return fmt.Errorf("unknown metric %q (cosine|dot|euclidean)", m)
	}
	ks := flagval(args, "k", "")
	if ks == "" {
		return fmt.Errorf("nth requires --k")
	}
	k, err := strconv.Atoi(ks)
	if err != nil || k <= 0 || k > len(c.Vecs) {
		return fmt.Errorf("--k must be between 1 and %d", len(c.Vecs))
	}
	hits := c.Rank(m, q)
	h := hits[k-1]
	fmt.Printf("%d\t%s\t%s\n", h.Rank, store.FormatF(h.Score), h.ID)
	return nil
}

func cmdInfo(args []string) error {
	pos := positional(args)
	if len(pos) != 1 {
		return fmt.Errorf("info takes exactly one corpus path")
	}
	c, err := store.Open(pos[0])
	if err != nil {
		return err
	}
	fmt.Printf("dims=%d count=%d seed=%d\n", c.Dims, len(c.Vecs), c.Seed)
	return nil
}

// queryLine is one parsed line of a batch query file.
type queryLine struct {
	Label string
	Q     []float64
}

func readQueryFile(path string, dims int) ([]queryLine, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []queryLine
	for ln, line := range strings.Split(string(data), "\n") {
		t := strings.TrimSpace(line)
		if t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		label := t
		rest := ""
		if i := strings.IndexAny(t, " \t"); i >= 0 {
			label, rest = t[:i], strings.TrimSpace(t[i+1:])
		}
		q, err := parseQuery(rest, dims)
		if err != nil {
			return nil, fmt.Errorf("%s line %d: %v", path, ln+1, err)
		}
		out = append(out, queryLine{Label: label, Q: q})
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s contains no queries", path)
	}
	return out, nil
}

func cmdBatch(args []string) error {
	pos := positional(args)
	if len(pos) != 1 {
		return fmt.Errorf("batch takes exactly one corpus path")
	}
	c, err := store.Open(pos[0])
	if err != nil {
		return err
	}
	file := flagval(args, "file", "")
	if file == "" {
		return fmt.Errorf("batch requires --file")
	}
	ql, err := readQueryFile(file, c.Dims)
	if err != nil {
		return err
	}
	m := store.Metric(flagval(args, "metric", "cosine"))
	if !store.ValidMetric(m) {
		return fmt.Errorf("unknown metric %q (cosine|dot|euclidean)", m)
	}
	k := 0
	if ks := flagval(args, "k", "0"); ks != "0" {
		if k, err = strconv.Atoi(ks); err != nil || k <= 0 {
			return fmt.Errorf("--k must be a positive integer")
		}
	}
	format := flagval(args, "format", "table")
	for _, one := range ql {
		hits := c.Rank(m, one.Q)
		if k > 0 {
			if k > len(hits) {
				return fmt.Errorf("query %q: --k %d exceeds corpus size %d", one.Label, k, len(hits))
			}
			hits = hits[:k]
		}
		switch format {
		case "json":
			var b strings.Builder
			fmt.Fprintf(&b, "{\"query\":%q,\"hits\":[", one.Label)
			for i, h := range hits {
				if i > 0 {
					b.WriteString(",")
				}
				fmt.Fprintf(&b, `{"rank":%d,"id":%q,"score":%s}`,
					h.Rank, h.ID, store.FormatF(h.Score))
			}
			b.WriteString("]}\n")
			fmt.Print(b.String())
		default:
			fmt.Printf("query %s\n", one.Label)
			printHits(format, hits)
		}
	}
	return nil
}

func cmdStat(args []string) error {
	pos := positional(args)
	if len(pos) != 1 {
		return fmt.Errorf("stat takes exactly one corpus path")
	}
	c, err := store.Open(pos[0])
	if err != nil {
		return err
	}
	bins, err := strconv.Atoi(flagval(args, "bins", "10"))
	if err != nil || bins <= 0 {
		return fmt.Errorf("--bins must be a positive integer")
	}
	stats := c.Stats()
	edges, counts := c.NormHistogram(bins)
	switch format := flagval(args, "format", "table"); format {
	case "csv":
		fmt.Println("dim,min,max,mean,std")
		for _, s := range stats {
			fmt.Printf("%d,%s,%s,%s,%s\n", s.Dim,
				store.FormatF(s.Min), store.FormatF(s.Max),
				store.FormatF(s.Mean), store.FormatF(s.Std))
		}
		fmt.Println("bin,edge_lo,edge_hi,count")
		for b := range counts {
			fmt.Printf("%d,%s,%s,%d\n", b, store.FormatF(edges[b]),
				store.FormatF(edges[b+1]), counts[b])
		}
	case "json":
		var b strings.Builder
		b.WriteString("{\"dims\":[")
		for i, s := range stats {
			if i > 0 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, `{"dim":%d,"min":%s,"max":%s,"mean":%s,"std":%s}`,
				s.Dim, store.FormatF(s.Min), store.FormatF(s.Max),
				store.FormatF(s.Mean), store.FormatF(s.Std))
		}
		b.WriteString("],\"norm_histogram\":{\"bins\":[")
		for i := range counts {
			if i > 0 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, `{"bin":%d,"edge_lo":%s,"edge_hi":%s,"count":%d}`,
				i, store.FormatF(edges[i]), store.FormatF(edges[i+1]), counts[i])
		}
		b.WriteString("]}}\n")
		fmt.Print(b.String())
	default:
		fmt.Println("dim\tmin\tmax\tmean\tstd")
		for _, s := range stats {
			fmt.Printf("%d\t%s\t%s\t%s\t%s\n", s.Dim,
				store.FormatF(s.Min), store.FormatF(s.Max),
				store.FormatF(s.Mean), store.FormatF(s.Std))
		}
		fmt.Println("bin\tedge_lo\tedge_hi\tcount")
		for b := range counts {
			fmt.Printf("%d\t%s\t%s\t%d\n", b, store.FormatF(edges[b]),
				store.FormatF(edges[b+1]), counts[b])
		}
	}
	return nil
}

func cmdQuantize(args []string) error {
	pos := positional(args)
	if len(pos) != 1 {
		return fmt.Errorf("quantize takes exactly one corpus path")
	}
	c, err := store.Open(pos[0])
	if err != nil {
		return err
	}
	q := c.Quantize()
	withErr := flagval(args, "error", "false") == "true"
	stats := c.Stats()
	switch format := flagval(args, "format", "table"); format {
	case "csv":
		fmt.Println("dim,scale,offset")
		for d := range q.Scales {
			fmt.Printf("%d,%s,%s\n", d, store.FormatF(q.Scales[d]), store.FormatF(q.Offsets[d]))
		}
	case "json":
		var b strings.Builder
		b.WriteString("{\"dims\":[")
		for d := range q.Scales {
			if d > 0 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, `{"dim":%d,"scale":%s,"offset":%s,"min":%s,"max":%s}`,
				d, store.FormatF(q.Scales[d]), store.FormatF(q.Offsets[d]),
				store.FormatF(stats[d].Min), store.FormatF(stats[d].Max))
		}
		b.WriteString("]}\n")
		fmt.Print(b.String())
	default:
		fmt.Println("dim\tscale\toffset")
		for d := range q.Scales {
			fmt.Printf("%d\t%s\t%s\n", d, store.FormatF(q.Scales[d]), store.FormatF(q.Offsets[d]))
		}
	}
	if withErr {
		maxAbs, meanAbs := c.ReconstructionError(q)
		fmt.Printf("reconstruction_error\tmax=%s\tmean=%s\n",
			store.FormatF(maxAbs), store.FormatF(meanAbs))
	}
	return nil
}

func cmdQRank(args []string) error {
	pos := positional(args)
	if len(pos) != 1 {
		return fmt.Errorf("qrank takes exactly one corpus path")
	}
	c, err := store.Open(pos[0])
	if err != nil {
		return err
	}
	q, err := parseQuery(flagval(args, "q", ""), c.Dims)
	if err != nil {
		return err
	}
	m := store.Metric(flagval(args, "metric", "cosine"))
	if !store.ValidMetric(m) {
		return fmt.Errorf("unknown metric %q (cosine|dot|euclidean)", m)
	}
	qz := c.Quantize()
	hits := qz.QRank(m, q)
	if k := flagval(args, "k", "0"); k != "0" {
		n, err := strconv.Atoi(k)
		if err != nil || n <= 0 || n > len(hits) {
			return fmt.Errorf("--k must be between 1 and %d", len(hits))
		}
		hits = hits[:n]
	}
	printHits(flagval(args, "format", "table"), hits)
	return nil
}
