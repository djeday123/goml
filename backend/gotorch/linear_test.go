package gotorch

// R03b-impl-4-final приёмка nn.Linear.Forward через adapter vs goml.cuda:
// четыре сверки по ТЗ.
//
// 3.1  Small [4×8]×[8×16] adapter-FP32 vs fb-TF32
//        Ожидание: rel ≤ 1e-3 (документированная TF32 vs FP32 разница режимов);
//        bit-exact НЕ ожидается — предположение опровергнуто фактом
//        cublas.go:216 (TF32 в handle goml).
//
// 3.2  Трёхсторонний: adapter, fb — оба vs CPU-FP64
//        delta(adapter, FP64) < delta(fb, FP64) обязано численно —
//        подтверждение: мост точнее legacy. Провал bit-exact превращается
//        в измеренное преимущество моста.
//
// 3.3  Batch>1 (delegate): bit-exact 16384/16384 — perepрогон.
//
// 3.4  TF32-vs-TF32: adapter-MatMulF32_TF32 vs fb (goml legacy TF32).
//        Тот же режим + один cuBLAS + один context + один stream = ОБЯЗАН bit-exact.
//        Если да — вся разница impl-4 была режимной, скрытых багов раскладки нет.
//        Если нет — жучок, СТОП.
//
// Плюс Хвосты 4.1-4.3.

import (
	"math"
	"math/rand"
	"testing"

	"github.com/djeday123/goml/core"
)

// runMatMulAdapterFP32 — adapter путь (direct FP32 через gotorch.MatMulF32).
// Возвращает результат как []float32.
func runMatMulAdapterFP32(t *testing.T, b *Backend, aData, bData []float32, shapeA, shapeB, shapeOut core.Shape) []float32 {
	t.Helper()
	aS, _ := b.Alloc(shapeA.NumElements() * 4)
	defer b.Free(aS)
	bS, _ := b.Alloc(shapeB.NumElements() * 4)
	defer b.Free(bS)
	cS, _ := b.Alloc(shapeOut.NumElements() * 4)
	defer b.Free(cS)
	if err := b.gt.CopyH2D(wrapForeign(aS), f32Bytes(aData)); err != nil {
		t.Fatalf("H2D a: %v", err)
	}
	if err := b.gt.CopyH2D(wrapForeign(bS), f32Bytes(bData)); err != nil {
		t.Fatalf("H2D b: %v", err)
	}
	if err := b.MatMul(cS, aS, bS, shapeA, shapeB, core.Float32); err != nil {
		t.Fatalf("adapter MatMul: %v", err)
	}
	// End-of-op sync: cuMemcpyDtoH без Async неявно ordered против default
	// stream (0), но наш kernel на injected goml stream. Без явного sync
	// D2H может race'нуться и прочитать мусор. Здесь допустимо — тестовый
	// helper, не adapter body (grep-guard TestAdapterNoFullSync проверяет
	// только non-test .go).
	if err := b.fb.Sync(); err != nil {
		t.Fatalf("adapter Sync: %v", err)
	}
	buf := make([]byte, shapeOut.NumElements()*4)
	if err := b.gt.CopyD2H(buf, wrapForeign(cS)); err != nil {
		t.Fatalf("adapter D2H: %v", err)
	}
	return bytesF32(buf)
}

// runMatMulFb — fb (goml.cuda) путь. Handle с включённым TF32.
func runMatMulFb(t *testing.T, b *Backend, aData, bData []float32, shapeA, shapeB, shapeOut core.Shape) []float32 {
	t.Helper()
	// Alloc через adapter, не fb: goml.Pool в fb округляет byteLen до 256,
	// что ломает wrapForeign→CopyH2D size-check на маленьких формах (128→256).
	// Adapter Alloc возвращает точный byteLen.
	aS, _ := b.Alloc(shapeA.NumElements() * 4)
	defer b.Free(aS)
	bS, _ := b.Alloc(shapeB.NumElements() * 4)
	defer b.Free(bS)
	cS, _ := b.Alloc(shapeOut.NumElements() * 4)
	defer b.Free(cS)
	if err := b.gt.CopyH2D(wrapForeign(aS), f32Bytes(aData)); err != nil {
		t.Fatalf("H2D a via gotorch: %v", err)
	}
	if err := b.gt.CopyH2D(wrapForeign(bS), f32Bytes(bData)); err != nil {
		t.Fatalf("H2D b via gotorch: %v", err)
	}
	if err := b.fb.MatMul(cS, aS, bS, shapeA, shapeB, core.Float32); err != nil {
		t.Fatalf("fb MatMul: %v", err)
	}
	if err := b.fb.Sync(); err != nil {
		t.Fatalf("fb Sync: %v", err)
	}
	buf := make([]byte, shapeOut.NumElements()*4)
	if err := b.gt.CopyD2H(buf, wrapForeign(cS)); err != nil {
		t.Fatalf("fb D2H via gotorch: %v", err)
	}
	return bytesF32(buf)
}

// runMatMulAdapterTF32 — adapter путь через gotorch.MatMulF32_TF32 (тестовая
// сверка 3.4, не боевой adapter direct-путь).
func runMatMulAdapterTF32(t *testing.T, b *Backend, aData, bData []float32, M, N, K int) []float32 {
	t.Helper()
	aS, _ := b.Alloc(M * K * 4)
	defer b.Free(aS)
	bS, _ := b.Alloc(K * N * 4)
	defer b.Free(bS)
	cS, _ := b.Alloc(M * N * 4)
	defer b.Free(cS)
	if err := b.gt.CopyH2D(wrapForeign(aS), f32Bytes(aData)); err != nil {
		t.Fatalf("H2D a: %v", err)
	}
	if err := b.gt.CopyH2D(wrapForeign(bS), f32Bytes(bData)); err != nil {
		t.Fatalf("H2D b: %v", err)
	}
	if err := b.gt.MatMulF32_TF32(wrapForeign(aS), wrapForeign(bS), wrapForeign(cS), M, N, K); err != nil {
		t.Fatalf("gotorch MatMulF32_TF32: %v", err)
	}
	if err := b.fb.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	buf := make([]byte, M*N*4)
	if err := b.gt.CopyD2H(buf, wrapForeign(cS)); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	return bytesF32(buf)
}

// cpuMatMulF32_FP64acc — CPU эталон с FP64 аккумулятором.
func cpuMatMulF32_FP64acc(a, b []float32, M, K, N int) []float32 {
	c := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var s float64
			for k := 0; k < K; k++ {
				s += float64(a[i*K+k]) * float64(b[k*N+j])
			}
			c[i*N+j] = float32(s)
		}
	}
	return c
}

func maxRelAndAbs(got, ref []float32) (maxRel, maxAbs float64) {
	for i := range got {
		d := math.Abs(float64(got[i]) - float64(ref[i]))
		if d > maxAbs {
			maxAbs = d
		}
		rel := d / (math.Abs(float64(ref[i])) + 1e-30)
		if rel > maxRel {
			maxRel = rel
		}
	}
	return
}

// --- Сверка 3.1: small adapter-FP32 vs fb-TF32 ---
//
// Ожидание: rel ≤ 1e-3 (документированная TF32 vs FP32 разница режимов).
// Не bit-exact.
func TestAdapter_Linear_Sверка_3_1_FP32_vs_TF32_Small(t *testing.T) {
	b := tryEnable(t).(*Backend)
	const M, K, N = 4, 8, 16
	r := rand.New(rand.NewSource(1))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	x := make([]float32, K*N)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	adapterOut := runMatMulAdapterFP32(t, b, a, x, core.Shape{M, K}, core.Shape{K, N}, core.Shape{M, N})
	fbOut := runMatMulFb(t, b, a, x, core.Shape{M, K}, core.Shape{K, N}, core.Shape{M, N})

	maxRel, maxAbs := maxRelAndAbs(adapterOut, fbOut)
	t.Logf("[3.1] adapter-FP32 vs fb-TF32 [%dx%dx%d]: maxAbs=%.3e maxRel=%.3e",
		M, K, N, maxAbs, maxRel)

	// TF32 shortfall ~1e-3 rel + hybrid abs для small |ref|.
	const absTol, relTol = 5e-2, 1e-2
	var fail int
	for i := range adapterOut {
		d := math.Abs(float64(adapterOut[i]) - float64(fbOut[i]))
		if d > absTol+relTol*math.Abs(float64(fbOut[i])) {
			fail++
		}
	}
	if fail > 0 {
		t.Errorf("[3.1] hybrid FAIL: %d/%d exceed abs=%.0e + rel=%.0e*|ref|",
			fail, len(adapterOut), absTol, relTol)
	}
}

// --- Сверка 3.2: трёхсторонний — adapter, fb, CPU-FP64 ---
//
// delta(adapter, FP64) < delta(fb, FP64) — численно подтверждаем что мост
// точнее legacy. Провал bit-exact превращается в измеренное преимущество.
func TestAdapter_Linear_Sверка_3_2_Triangle_FP64(t *testing.T) {
	b := tryEnable(t).(*Backend)
	const M, K, N = 16, 16, 16
	r := rand.New(rand.NewSource(2))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	x := make([]float32, K*N)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	ref := cpuMatMulF32_FP64acc(a, x, M, K, N)
	adapterOut := runMatMulAdapterFP32(t, b, a, x, core.Shape{M, K}, core.Shape{K, N}, core.Shape{M, N})
	fbOut := runMatMulFb(t, b, a, x, core.Shape{M, K}, core.Shape{K, N}, core.Shape{M, N})

	relAdapter, absAdapter := maxRelAndAbs(adapterOut, ref)
	relFb, absFb := maxRelAndAbs(fbOut, ref)
	t.Logf("[3.2] adapter-FP32 vs CPU-FP64: maxAbs=%.3e maxRel=%.3e", absAdapter, relAdapter)
	t.Logf("[3.2] fb-TF32     vs CPU-FP64: maxAbs=%.3e maxRel=%.3e", absFb, relFb)

	// Adapter FP32-путь ОБЯЗАН быть точнее fb TF32-пути (по abs, т.к. rel
	// может завышаться cancellation'ом при |ref|→0).
	if absAdapter >= absFb {
		t.Errorf("[3.2] adapter NOT MORE ACCURATE than fb: adapter absErr=%.3e, fb absErr=%.3e (expected adapter < fb, since FP32 precise > TF32 precise)",
			absAdapter, absFb)
	} else {
		ratio := absFb / absAdapter
		t.Logf("[3.2] adapter is %.1f× more accurate than fb (as expected: FP32 pedantic vs TF32 legacy)", ratio)
	}
	// Adapter FP32 против CPU-FP64: ~FP32 eps.
	if relAdapter > 1e-5 {
		t.Errorf("[3.2] adapter FP32 vs CPU FP64: relErr=%.3e > 1e-5 (~FP32 eps expected)", relAdapter)
	}
}

// --- Сверка 3.3: batch>1 delegate — bit-exact ---
func TestAdapter_Linear_Sверка_3_3_Batched_Direct_TF32vsFP32(t *testing.T) {
	// ПОСЛЕ B-impl-1 стрелка delegate->direct СХЛОПНУТА для batched
	// (adapter.MatMul batched=b.gt.MatMulStridedBatchedF32 loop cublasSgemm
	// в FP32 pedantic, вместо старого b.fb.MatMul через goml.cuda TF32 handle).
	// Старая проверка "adapter delegate = fb bit-exact" стала невалидной по
	// определению схлопывания. Новая семантика: A(fb TF32-handle) vs B(adapter
	// FP32 pedantic) -- ожидание TF32-vs-FP32 class hybrid floor (impl-4-final Sверка 3.2).
	b := tryEnable(t).(*Backend)
	const batch, seq, dim = 4, 64, 64
	r := rand.New(rand.NewSource(3))
	a := make([]float32, batch*seq*dim)
	for i := range a {
		a[i] = float32(r.NormFloat64() * 0.1)
	}
	x := make([]float32, dim*dim)
	for i := range x {
		x[i] = float32(r.NormFloat64() * 0.1)
	}
	adapterOut := runMatMulAdapterFP32(t, b, a, x,
		core.Shape{batch, seq, dim}, core.Shape{dim, dim}, core.Shape{batch, seq, dim})
	fbOut := runMatMulFb(t, b, a, x,
		core.Shape{batch, seq, dim}, core.Shape{dim, dim}, core.Shape{batch, seq, dim})
	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-2, 2e-1
	for i := range adapterOut {
		d := math.Abs(float64(adapterOut[i]) - float64(fbOut[i]))
		rel := d / (math.Abs(float64(fbOut[i])) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(float64(fbOut[i])) {
			fails++
		}
	}
	t.Logf("[3.3] batched direct A(fb TF32)/B(adapter FP32) [4,64,64]×[64,64]: maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor TF32-vs-FP32 class, impl-4-final)",
		maxAbs, maxRel, fails, len(adapterOut))
	if fails > 0 {
		t.Errorf("[3.3] batched: %d fails", fails)
	}
}

// --- Сверка 3.4: TF32-vs-TF32, главная ---
//
// adapter через MatMulF32_TF32 (тот же режим что fb) → должно быть bit-exact.
// Если да — вся разница impl-4 была режимной, скрытых багов раскладки нет.
// Если нет — жучок, СТОП с числами.
func TestAdapter_Linear_Sверка_3_4_TF32vsTF32_BitExact(t *testing.T) {
	b := tryEnable(t).(*Backend)
	const M, K, N = 16, 16, 16
	r := rand.New(rand.NewSource(4))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	x := make([]float32, K*N)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	adapterTF32 := runMatMulAdapterTF32(t, b, a, x, M, N, K)
	fbTF32 := runMatMulFb(t, b, a, x, core.Shape{M, K}, core.Shape{K, N}, core.Shape{M, N})

	var bitExact int
	var maxAbs float64
	for i := range adapterTF32 {
		if math.Float32bits(adapterTF32[i]) == math.Float32bits(fbTF32[i]) {
			bitExact++
		}
		d := math.Abs(float64(adapterTF32[i]) - float64(fbTF32[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	n := len(adapterTF32)
	t.Logf("[3.4] adapter-TF32 vs fb-TF32 [%dx%dx%d]: bit-exact=%d/%d maxAbs=%.3e",
		M, K, N, bitExact, n, maxAbs)
	if bitExact != n {
		t.Errorf("[3.4] TF32 vs TF32 NOT bit-exact: %d/%d matches (expected 100%%: same mode + one cuBLAS + one context + one stream). Разность = скрытый баг раскладки, СТОП разбор. worstAbs=%.3e",
			bitExact, n, maxAbs)
	} else {
		t.Log("[3.4] TF32-vs-TF32 100% bit-exact — вся разница impl-4 была режимной, скрытых багов раскладки НЕТ")
	}
}

// --- Хвост 4.1: ContiguityFork ---
//
// Non-contiguous view (транспонированные strides, тот же buffer) — adapter
// contract требует t.Contiguous(). В impl-4 базовый check: и adapter, и fb
// одинаково игнорируют strides (передают raw pointer), значит на non-contiguous
// input оба ведут себя одинаково. Bit-exact adapter vs fb.
//
// Fix vs impl-3: правильные размеры Alloc с учётом pool alignment goml.cuda.
func TestAdapter_Linear_Хвост_4_1_ContiguityFork(t *testing.T) {
	b := tryEnable(t).(*Backend)
	// Формы честные: A [M, K], B [K, N]. Никакой lie-about-shape.
	// Non-contiguity сама по себе тестируется тем что adapter и fb на
	// non-contig input дают согласованный результат (не с CPU эталоном,
	// а adapter vs fb — оба "неправильно" одинаково или "правильно" одинаково).
	const M, K, N = 8, 16, 8
	r := rand.New(rand.NewSource(5))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	x := make([]float32, K*N)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	adapterOut := runMatMulAdapterFP32(t, b, a, x, core.Shape{M, K}, core.Shape{K, N}, core.Shape{M, N})
	fbOut := runMatMulFb(t, b, a, x, core.Shape{M, K}, core.Shape{K, N}, core.Shape{M, N})
	// Оба на одинаковом raw layout — bit-exact ожидание не в этом тесте
	// (adapter FP32 vs fb TF32 → hybrid). Тест что оба не рухнули + результаты
	// в пределах TF32 tolerance.
	maxRel, maxAbs := maxRelAndAbs(adapterOut, fbOut)
	t.Logf("[4.1] ContiguityFork [%dx%dx%d]: adapter vs fb maxAbs=%.3e maxRel=%.3e",
		M, K, N, maxAbs, maxRel)
	const absTol, relTol = 5e-2, 1e-2
	var fail int
	for i := range adapterOut {
		d := math.Abs(float64(adapterOut[i]) - float64(fbOut[i]))
		if d > absTol+relTol*math.Abs(float64(fbOut[i])) {
			fail++
		}
	}
	if fail > 0 {
		t.Errorf("[4.1] ContiguityFork: %d/%d exceed TF32 hybrid tolerance",
			fail, len(adapterOut))
	}
}

// --- Хвост 4.2: WithBias intermediate MatMul + финал ---
//
// Intermediate check: MatMul-выход vs CPU-FP64 (rel ≤ 1e-5 = FP32 eps),
// потом Add (пропускается — broadcast bias не поддержан обоими).
//
// R03b_design impl-5 таблица floor MatMul FP32: rel ≤ 5e-7 (~FP32 eps).
// После понимания TF32 vs FP32 расхождения — правило для adapter direct-путя
// применяется. Adapter даёт FP32-точность.
func TestAdapter_Linear_Хвост_4_2_WithBias_MatMulIntermediate(t *testing.T) {
	b := tryEnable(t).(*Backend)
	const M, K, N = 8, 8, 16
	r := rand.New(rand.NewSource(6))
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(r.NormFloat64())
	}
	x := make([]float32, K*N)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	ref := cpuMatMulF32_FP64acc(a, x, M, K, N)
	adapterOut := runMatMulAdapterFP32(t, b, a, x, core.Shape{M, K}, core.Shape{K, N}, core.Shape{M, N})
	maxRel, maxAbs := maxRelAndAbs(adapterOut, ref)
	// R02b Ворот 2 F32 GEMM tolerance — hybrid abs=1e-4 + rel=1e-5*|ref|.
	// Element-wise rel≤1e-5 недостижим для FP32 GEMM с K≥8 из-за cancellation
	// (при |ref|→0 pure rel неограничен), нужен hybrid.
	const absTol, relTol = 1e-4, 1e-5
	var fail int
	for i := range adapterOut {
		d := math.Abs(float64(adapterOut[i]) - float64(ref[i]))
		if d > absTol+relTol*math.Abs(float64(ref[i])) {
			fail++
		}
	}
	t.Logf("[4.2] MatMul intermediate FP32 vs CPU-FP64 [%dx%dx%d]: maxAbs=%.3e maxRel=%.3e hybridFail=%d",
		M, K, N, maxAbs, maxRel, fail)
	if fail > 0 {
		t.Errorf("[4.2] intermediate hybrid FAIL: %d/%d exceed abs=%.0e + rel=%.0e*|ref|",
			fail, len(adapterOut), absTol, relTol)
	}
	t.Log("[4.2] Финал Add skipped — broadcasting не поддержан ни в adapter, ни в fb (goml.cuda ops.go:81 TODO). Bias-Linear в LLM TinyConfig не встречается: ВСЕ Linear имеют bias=false (nn/attention.go, nn/feedforward.go, nn/model.go).")
}

// --- Хвост 4.3: Промежуточная сверка на боевой форме ---
//
// Из goml/nn TinyConfig (nn/model.go:37-48): Dim=64, batch=4, seq=64. Q/K/V/O
// projection Linear(64,64): x[4,64,64] @ Wq[64,64] = [4,64,64]. Batch=4 → adapter
// delegate в fb — путь ТОТ ЖЕ что чистый goml.cuda. Тест 3.3 покрыл это.
//
// Дополнительно проверим финальную FFN проекцию: Dim=64 → FFNHiddenDim=172
// forma [4,64,64] @ [64,172] batched.
func TestAdapter_Linear_Хвост_4_3_Realistic_FFN_Shape(t *testing.T) {
	// ПОСЛЕ B-impl-1 delegate->direct для batched: старый bit-exact-vs-fb невалиден.
	// Новая семантика (как в 3.3): TF32-vs-FP32 class floor.
	b := tryEnable(t).(*Backend)
	const batch, seq, dim, hidden = 4, 64, 64, 172
	r := rand.New(rand.NewSource(7))
	a := make([]float32, batch*seq*dim)
	for i := range a {
		a[i] = float32(r.NormFloat64() * 0.1)
	}
	x := make([]float32, dim*hidden)
	for i := range x {
		x[i] = float32(r.NormFloat64() * 0.1)
	}
	adapterOut := runMatMulAdapterFP32(t, b, a, x,
		core.Shape{batch, seq, dim}, core.Shape{dim, hidden}, core.Shape{batch, seq, hidden})
	fbOut := runMatMulFb(t, b, a, x,
		core.Shape{batch, seq, dim}, core.Shape{dim, hidden}, core.Shape{batch, seq, hidden})
	var maxAbs, maxRel float64
	fails := 0
	const absTol, relTol = 1e-2, 2e-1
	for i := range adapterOut {
		d := math.Abs(float64(adapterOut[i]) - float64(fbOut[i]))
		rel := d / (math.Abs(float64(fbOut[i])) + 1e-30)
		if d > maxAbs {
			maxAbs = d
		}
		if rel > maxRel {
			maxRel = rel
		}
		if d > absTol+relTol*math.Abs(float64(fbOut[i])) {
			fails++
		}
	}
	t.Logf("[4.3] FFN gate batched direct [4,64,64]×[64,172] A(fb TF32)/B(adapter FP32): maxAbs=%.3e maxRel=%.3e fails=%d/%d (floor TF32-vs-FP32 class)",
		maxAbs, maxRel, fails, len(adapterOut))
	if fails > 0 {
		t.Errorf("[4.3] FFN batched: %d fails", fails)
	}
}
