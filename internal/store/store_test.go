package store

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestGenerateDeterministic(t *testing.T) {
	a := Generate(42, 50, 4)
	b := Generate(42, 50, 4)
	if len(a.Vecs) != 50 || a.Dims != 4 {
		t.Fatalf("bad shape: %+v", a)
	}
	for i := range a.Vecs {
		if a.Vecs[i].ID != b.Vecs[i].ID {
			t.Fatalf("id mismatch at %d", i)
		}
		for j := range a.Vecs[i].Comps {
			if a.Vecs[i].Comps[j] != b.Vecs[i].Comps[j] {
				t.Fatalf("component mismatch at %d,%d", i, j)
			}
		}
	}
}

func TestRoundTrip(t *testing.T) {
	c := Generate(7, 10, 3)
	p := filepath.Join(t.TempDir(), "c.txt")
	if err := c.Save(p); err != nil {
		t.Fatal(err)
	}
	c2, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(c2.Vecs) != len(c.Vecs) || c2.Dims != c.Dims || c2.Seed != c.Seed {
		t.Fatalf("round-trip shape mismatch")
	}
	// Save is canonical: a re-save is byte-identical.
	p2 := filepath.Join(t.TempDir(), "c2.txt")
	if err := c2.Save(p2); err != nil {
		t.Fatal(err)
	}
	x1, _ := os.ReadFile(p)
	x2, _ := os.ReadFile(p2)
	if string(x1) != string(x2) {
		t.Fatalf("canonical save not byte-stable")
	}
}

func TestRankTieBreak(t *testing.T) {
	c := &Corpus{Dims: 1, index: map[string]int{}}
	// identical scores: tie broken by id ascending
	c.Append("b", []float64{1})
	c.Append("a", []float64{1})
	c.Append("c", []float64{1})
	hits := c.Rank(Dot, []float64{1})
	if hits[0].ID != "a" || hits[1].ID != "b" || hits[2].ID != "c" {
		t.Fatalf("tie-break failed: %v", hits)
	}
	if hits[0].Rank != 1 || hits[2].Rank != 3 {
		t.Fatalf("ranks not 1-based: %v", hits)
	}
}

func TestCosineZeroNorm(t *testing.T) {
	if got := Score(Cosine, []float64{0, 0}, []float64{1, 1}); got != 0 {
		t.Fatalf("zero-norm cosine should be 0, got %v", got)
	}
}

func TestEuclideanNegated(t *testing.T) {
	got := Score(Euclidean, []float64{0, 0}, []float64{3, 4})
	if math.Abs(got-(-5)) > 1e-12 {
		t.Fatalf("euclidean should be negated distance, got %v", got)
	}
}
