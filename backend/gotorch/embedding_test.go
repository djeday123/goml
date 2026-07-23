package gotorch

// P3-EMB Stage 4: adapter direct EmbeddingF32/F64 fwd+bwd против F64 CPU-судьи.
// Дополнительно: A/B forward (goml.cuda.Embedding с int64 vs наш adapter F32 с int32).
//
// Форма: LLaMA-tiny-ish [vocab=32000, hidden=256, n=64] по default LLM Config.
//
// Прогнозы (pre-registered):
//   A/B fwd: bit-exact (gather = memcpy строки; int64 vs int32 через равные значения).
//   B/J fwd F32: bit-exact vs F64 CPU ref (single-precision gather без потерь).
//   B/J fwd F64: bit-exact.
//   B/J bwd F32 (без коллизий): bit-exact (нет race).
//   B/J bwd F32 (с коллизиями): hybrid abs=1e-4+rel=1e-5 (atomic-order шум).

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

func i32BytesGoml(v []int32) []byte {
	buf := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(x))
	}
	return buf
}

func i64BytesGoml(v []int64) []byte {
	buf := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(buf[i*8:], uint64(x))
	}
	return buf
}

// allocIndicesInt32 — upload int32 indices через adapter (CopyH2D байты).
func allocIndicesInt32(t *testing.T, b backend.Backend, xs []int32) backend.Storage {
	t.Helper()
	cpuB, _ := backend.Get(backend.CPU)
	cpu, err := cpuB.Alloc(len(xs) * 4)
	if err != nil {
		t.Fatalf("cpu.Alloc: %v", err)
	}
	copy(cpu.Bytes(), i32BytesGoml(xs))
	gpu, err := b.ToDevice(backend.CUDADevice(0), cpu)
	cpuB.Free(cpu)
	if err != nil {
		t.Fatalf("ToDevice indices: %v", err)
	}
	return gpu
}

func allocIndicesInt64(t *testing.T, b backend.Backend, xs []int64) backend.Storage {
	t.Helper()
	cpuB, _ := backend.Get(backend.CPU)
	cpu, err := cpuB.Alloc(len(xs) * 8)
	if err != nil {
		t.Fatalf("cpu.Alloc: %v", err)
	}
	copy(cpu.Bytes(), i64BytesGoml(xs))
	gpu, err := b.ToDevice(backend.CUDADevice(0), cpu)
	cpuB.Free(cpu)
	if err != nil {
		t.Fatalf("ToDevice indices64: %v", err)
	}
	return gpu
}

func TestAdapterEmbeddingF32_BvsJ(t *testing.T) {
	// Pre-registered: fwd bit-exact.
	bAny := tryEnable(t)
	b, ok := bAny.(*Backend)
	if !ok {
		t.Fatalf("adapter Get(CUDA) returned %T, want *Backend", bAny)
	}

	const vocab, hidden, n = 32000, 256, 64
	r := rand.New(rand.NewSource(2026))
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = float32(r.NormFloat64())
	}
	indices := make([]int32, n)
	for i := range indices {
		indices[i] = int32(r.Intn(vocab))
	}
	// F64 judge (via F32 table promoted).
	tableF64 := make([]float64, len(table))
	for i, v := range table {
		tableF64[i] = float64(v)
	}
	refOut := make([]float64, n*hidden)
	for i := 0; i < n; i++ {
		idx := int(indices[i])
		for d := 0; d < hidden; d++ {
			refOut[i*hidden+d] = tableF64[idx*hidden+d]
		}
	}

	tS := allocFromSlice(t, b, table)
	defer b.Free(tS)
	iS := allocIndicesInt32(t, b, indices)
	defer b.Free(iS)
	oS, _ := b.Alloc(n * hidden * 4)
	defer b.Free(oS)

	if err := b.EmbeddingF32(tS, iS, oS, vocab, hidden, n); err != nil {
		t.Fatalf("adapter EmbeddingF32: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := downloadF32(t, b, oS)

	mismatches := 0
	for i := range got {
		if float64(got[i]) != refOut[i] {
			mismatches++
		}
	}
	t.Logf("B(adapter F32) vs J(F64) fwd [v=%d h=%d n=%d]: bit-exact=%d/%d", vocab, hidden, n, len(got)-mismatches, len(got))
	if mismatches > 0 {
		t.Errorf("adapter EmbeddingF32 vs judge: %d mismatches (fwd прогноз bit-exact)", mismatches)
	}
}

// A/B fwd: goml.cuda (int64) vs adapter (int32) — output bit-exact.
// Проверяем что вкат разных dtype индексов даёт один output (при равных значениях индексов).
func TestAdapterEmbedding_AvsB_Forward(t *testing.T) {
	// Baseline (A) через goml.cuda.
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if gomlB.Name() == "gotorch-adapter" {
		t.Skipf("adapter уже включён — этот тест требует чистого goml.cuda пути; порядок test suite (skip)")
	}

	const vocab, hidden, n = 256, 64, 16 // gputrain shape
	r := rand.New(rand.NewSource(3030))
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = float32(r.NormFloat64())
	}
	indices32 := make([]int32, n)
	indices64 := make([]int64, n)
	for i := range indices32 {
		indices32[i] = int32(r.Intn(vocab))
		indices64[i] = int64(indices32[i])
	}

	// Path A: goml.cuda с int64 indices.
	tS_A := allocFromSlice(t, gomlB, table)
	defer gomlB.Free(tS_A)
	iS_A := allocIndicesInt64(t, gomlB, indices64)
	defer gomlB.Free(iS_A)
	oS_A, _ := gomlB.Alloc(n * hidden * 4)
	defer gomlB.Free(oS_A)
	if err := gomlB.Embedding(oS_A, tS_A, iS_A, vocab, hidden, n, core.Float32); err != nil {
		t.Fatalf("goml.cuda.Embedding: %v", err)
	}
	// Enforce sync через probe (goml.cuda.Backend имеет Sync).
	if syncer, ok := gomlB.(interface{ Sync() error }); ok {
		syncer.Sync()
	}
	gotA := downloadF32(t, gomlB, oS_A)

	// Path B: adapter (int32).
	if err := Enable(); err != nil {
		t.Fatalf("adapter Enable: %v", err)
	}
	adapterAny, _ := backend.Get(backend.CUDA)
	adapter, ok := adapterAny.(*Backend)
	if !ok {
		t.Fatalf("adapter type-assert: %T", adapterAny)
	}
	tS_B := allocFromSlice(t, adapter, table)
	defer adapter.Free(tS_B)
	iS_B := allocIndicesInt32(t, adapter, indices32)
	defer adapter.Free(iS_B)
	oS_B, _ := adapter.Alloc(n * hidden * 4)
	defer adapter.Free(oS_B)
	if err := adapter.EmbeddingF32(tS_B, iS_B, oS_B, vocab, hidden, n); err != nil {
		t.Fatalf("adapter EmbeddingF32: %v", err)
	}
	adapter.Sync()
	gotB := downloadF32(t, adapter, oS_B)

	mismatches := 0
	for i := range gotA {
		if math.Float32bits(gotA[i]) != math.Float32bits(gotB[i]) {
			mismatches++
		}
	}
	t.Logf("A(goml.cuda int64) vs B(adapter int32) fwd [v=%d h=%d n=%d]: bit-exact=%d/%d", vocab, hidden, n, len(gotA)-mismatches, len(gotA))
	if mismatches > 0 {
		t.Errorf("A vs B fwd: %d bit-mismatches (прогноз bit-exact, разница только в dtype индексов при равных значениях)", mismatches)
	}
}

// TestAdapterEmbedding_AvsB_via_Interface — стрелка delegate->direct через
// backend.Backend.Embedding интерфейс. goml подаёт int64 tokens как есть,
// adapter internal cvt в scratch и вызов gt.EmbeddingF32I64. Прогноз:
// bit-exact (gather = memcpy row; cvt.u32.u64 не теряет данные при idx<2^31).
func TestAdapterEmbedding_AvsB_via_Interface(t *testing.T) {
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if gomlB.Name() == "gotorch-adapter" {
		t.Skipf("adapter уже включён — тест требует чистого goml.cuda (skip)")
	}

	const vocab, hidden, n = 256, 64, 16
	r := rand.New(rand.NewSource(5252))
	table := make([]float32, vocab*hidden)
	for i := range table {
		table[i] = float32(r.NormFloat64())
	}
	indices64 := make([]int64, n)
	for i := range indices64 {
		indices64[i] = int64(r.Intn(vocab))
	}

	// Path A: goml.cuda.Embedding через backend interface (int64 native).
	tS_A := allocFromSlice(t, gomlB, table)
	defer gomlB.Free(tS_A)
	iS_A := allocIndicesInt64(t, gomlB, indices64)
	defer gomlB.Free(iS_A)
	oS_A, _ := gomlB.Alloc(n * hidden * 4)
	defer gomlB.Free(oS_A)
	if err := gomlB.Embedding(oS_A, tS_A, iS_A, vocab, hidden, n, core.Float32); err != nil {
		t.Fatalf("goml.cuda.Embedding: %v", err)
	}
	if syncer, ok := gomlB.(interface{ Sync() error }); ok {
		syncer.Sync()
	}
	gotA := downloadF32(t, gomlB, oS_A)

	// Path B: adapter.Embedding через backend interface (delegate->direct через I64-фасад).
	if err := Enable(); err != nil {
		t.Fatalf("adapter Enable: %v", err)
	}
	adapterAny, _ := backend.Get(backend.CUDA)
	adapter, ok := adapterAny.(*Backend)
	if !ok {
		t.Fatalf("adapter type-assert: %T", adapterAny)
	}
	tS_B := allocFromSlice(t, adapter, table)
	defer adapter.Free(tS_B)
	iS_B := allocIndicesInt64(t, adapter, indices64)
	defer adapter.Free(iS_B)
	oS_B, _ := adapter.Alloc(n * hidden * 4)
	defer adapter.Free(oS_B)
	if err := adapter.Embedding(oS_B, tS_B, iS_B, vocab, hidden, n, core.Float32); err != nil {
		t.Fatalf("adapter.Embedding: %v", err)
	}
	adapter.Sync()
	gotB := downloadF32(t, adapter, oS_B)

	mismatches := 0
	for i := range gotA {
		if math.Float32bits(gotA[i]) != math.Float32bits(gotB[i]) {
			mismatches++
		}
	}
	t.Logf("A(goml.cuda int64) vs B(adapter int64->int32 фасад) fwd [v=%d h=%d n=%d]: bit-exact=%d/%d",
		vocab, hidden, n, len(gotA)-mismatches, len(gotA))
	if mismatches > 0 {
		t.Errorf("adapter.Embedding interface arrow: %d bit-mismatches (прогноз P1 bit-exact)", mismatches)
	}
}

func TestAdapterEmbeddingGradF32_BvsJ(t *testing.T) {
	// Pre-registered floor: hybrid abs=1e-4 + rel=1e-5 (соответствует F32 grad tests).
	bAny := tryEnable(t)
	b, ok := bAny.(*Backend)
	if !ok {
		t.Fatalf("adapter Get(CUDA): %T", bAny)
	}

	const vocab, hidden, n = 256, 64, 16 // gputrain-shape
	r := rand.New(rand.NewSource(4040))
	indices := make([]int32, n)
	for i := range indices {
		indices[i] = int32(r.Intn(vocab / 8)) // компрессия -> коллизии
	}
	dout := make([]float32, n*hidden)
	doutF64 := make([]float64, len(dout))
	for i := range dout {
		dout[i] = float32(r.NormFloat64())
		doutF64[i] = float64(dout[i])
	}
	// F64 CPU ref (детерминированный).
	refDt := make([]float64, vocab*hidden)
	for i := 0; i < n; i++ {
		idx := int(indices[i])
		for d := 0; d < hidden; d++ {
			refDt[idx*hidden+d] += doutF64[i*hidden+d]
		}
	}

	iS := allocIndicesInt32(t, b, indices)
	defer b.Free(iS)
	dS := allocFromSlice(t, b, dout)
	defer b.Free(dS)
	dtS, _ := b.Alloc(vocab * hidden * 4)
	defer b.Free(dtS)

	if err := b.EmbeddingGradF32(iS, dS, dtS, vocab, hidden, n); err != nil {
		t.Fatalf("adapter EmbeddingGradF32: %v", err)
	}
	if err := b.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	got := downloadF32(t, b, dtS)

	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-4, 1e-5
	for i := range got {
		d := math.Abs(float64(got[i]) - refDt[i])
		rel := d / (math.Abs(refDt[i]) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(refDt[i]) {
			fails++
		}
	}
	t.Logf("B(adapter F32) vs J(F64) bwd [v=%d h=%d n=%d compression=8]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor abs=%.0e+rel=%.0e·|ref|)",
		vocab, hidden, n, maxAbs, maxRel, fails, len(got), absTol, relTol)
	if fails > 0 {
		t.Errorf("adapter EmbeddingGradF32 vs judge: %d fails", fails)
	}
}
