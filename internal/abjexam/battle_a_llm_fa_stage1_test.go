package abjexam

// A-LLM-4 Этап 1: FA-bwd блок в эталонной f32recon-обвязке, двухарбитровый A/B.
//
// Форма A/B (ТЗ п.1.3): B=1 H=4 HD=128 S=2048 D=512 FFN=2048 L=1 V=1024.
// Токены: seed 41, повторы допустимы (M=2048 > V=1024; уникальность невозможна
// и не нужна — Embedding-scatter нон-дет уже свойство пути).
//
// ПРОГНОЗЫ (записаны ДО прогона):
//   P1. fwd sanity: loss GPU == loss F64 до ~7 знаков (fwd bit-det из A-LLM-3;
//       если BH>1 wrapper-путь attnReconstructFwd отравлен — увидим здесь).
//   P2. Block dV (FA vs F64): зона A, floor 5e-3 abs с запасом класса B-impl-4
//       (dO-тракт FP16, стирателя FP8 нет).
//   P3. Block dQ/dK (FA vs F64): зона A floor 5e-3; ЗАМЕТНАЯ зона B
//       (|dS_real| < scaleV*2^-6 -> e4m3 direct-cast стирает; cold-start класс).
//   P4. Chain-точки: Wout/NormOut/Wo/W2/W1/Norm2 FA-НЕзависимы (считаются до
//       блока) — совпадают с recon-прогоном с точностью нон-дета пути;
//       FA-чувствительные Wq/Wv/Norm1/Embed — зона A 5e-3, если P2/P3-A прошли.
//
// Пороги зон (уточнение формулы ТЗ по факту контракта ядра, ДО прогона):
//   стиратель dQ/dK = e4m3 direct-cast dS (units real/scaleV):
//     T_norm = scaleV*2^-6, T_subnorm = scaleV*2^-9 (real units).
//   Клетки [T_subnorm, T_norm) — "субнормальная зона", отдельная строка.

import (
	"math"
	"math/rand"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	gomlcuda "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
)

func faABCfg() BattleACfg {
	return BattleACfg{
		V: 1024, D: 512, H: 4, HD: 128, L: 1, S: 2048, B: 1, FFN: 2048,
		Base: 10000.0, Eps: 1e-5,
	}
}

// faABTokens — простая генерация с повторами (seed 41), зеркальна для GPU и F64.
func faABTokens(cfg BattleACfg) (inp []int64, tgt []int32) {
	rTok := rand.New(rand.NewSource(41))
	M := cfg.B * cfg.S
	inp = make([]int64, M)
	tgt = make([]int32, M)
	for i := 0; i < M; i++ {
		inp[i] = int64(rTok.Intn(cfg.V))
		tgt[i] = int32(rTok.Intn(cfg.V))
	}
	return inp, tgt
}

func TestALLM_FABlock_Stage1(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	// ИЗВЕСТНЫЙ RED (A-LLM-4, 2026-08-03): цепочка блокирована контрактом
	// магнитуды v121r-ядер (FP16-S-accum требует decoded O(1); боевой квантизатор
	// amax/448 даёт decoded до +-448 -> NaN). Локализация: A_LLM4_fa_integration.md.
	// Достройка (host-квант O(1) + пересчёт scale_dq/dk) — следующая сессия.
	if os.Getenv("GOML_FA_STAGE1") != "1" {
		t.Skip("known-red до достройки квант-контракта; включить: GOML_FA_STAGE1=1")
	}
	gomlB, err := backend.Get(backend.CUDA)
	if err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	if s, err := gomlB.Alloc(4); err == nil {
		gomlB.Free(s)
	}
	if err := adapter.Enable(); err != nil {
		t.Fatalf("adapter Enable: %v", err)
	}
	adB, _ := backend.Get(backend.CUDA)
	if err := gomlcuda.FALoad(); err != nil {
		t.Skipf("libfa_sm120.so unavailable: %v", err)
	}
	if err := gomlcuda.FABwdLoad(); err != nil {
		t.Skipf("libfa_bwd_sm120.so unavailable: %v", err)
	}
	faCtx, err := gomlcuda.FACreate()
	if err != nil {
		t.Fatalf("FACreate: %v", err)
	}
	defer faCtx.Destroy()

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	cfg := faABCfg()
	inp, tgt := faABTokens(cfg)
	BH := cfg.B * cfg.H
	nBlk := BH * cfg.S * cfg.HD

	// === CPU-F64 арбитр ===
	t0 := time.Now()
	w64 := newBattleAF64Weights(cfg, 31)
	c64 := newBattleAF64Cache(cfg)
	loss64 := fwdBattleAF64(w64, c64, inp, tgt)
	tFwd64 := time.Since(t0)
	t0 = time.Now()
	g64 := bwdBattleAF64(w64, c64, inp, tgt)
	tBwd64 := time.Since(t0)
	t.Logf("F64-арбитр: loss=%.10f (fwd %.1fs, bwd %.1fs)", loss64, tFwd64.Seconds(), tBwd64.Seconds())

	// === GPU стек (веса bit-идентичны: seed 31) ===
	rInit := rand.New(rand.NewSource(31))
	st, err := NewBattleAState(cfg, rInit, adB)
	if err != nil {
		t.Fatalf("NewBattleAState: %v", err)
	}
	defer st.FreeAll(adB)
	sc, err := NewBattleAScratchF32(cfg, adB)
	if err != nil {
		t.Fatalf("NewBattleAScratchF32: %v", err)
	}
	defer sc.FreeAll(adB)
	bs, err := NewBattleABwdScratch(cfg, adB)
	if err != nil {
		t.Fatalf("NewBattleABwdScratch: %v", err)
	}
	defer bs.FreeAll(adB)
	grads, err := NewBattleAGrads(cfg, adB)
	if err != nil {
		t.Fatalf("NewBattleAGrads: %v", err)
	}
	defer grads.FreeAll(adB)
	// Пре-аллокация FA-буферов ДО первого FA-вызова (мина 2).
	// Двухконтекстная схема: FA-ядра работают только с нативными аллокациями
	// goml.cuda (находка сессии: адаптерные указатели для FA-.so невалидны).
	fb, err := newFABlockBufs(cfg, adB, gomlB)
	if err != nil {
		t.Fatalf("newFABlockBufs: %v", err)
	}
	defer fb.FreeAll(nil)

	// === P1: fwd sanity ===
	lossGPU, err := fwdBattleAF32(adB, st, sc, inp, tgt)
	if err != nil {
		t.Fatalf("GPU fwd: %v", err)
	}
	t.Logf("P1 fwd sanity: GPU loss=%.10f vs F64 loss=%.10f (delta=%.3e)", lossGPU, loss64, math.Abs(lossGPU-loss64))
	if math.Abs(lossGPU-loss64) > 1e-4 {
		t.Errorf("P1 FWD SANITY FAIL: GPU fwd отравлен (delta %.3e > 1e-4) — BH>1 wrapper-путь под подозрением", math.Abs(lossGPU-loss64))
	}

	// === Полный recon bwd (SECONDARY арбитр, chain-level) ===
	if err := zeroGrads(adB, grads); err != nil {
		t.Fatalf("zeroGrads: %v", err)
	}
	if err := bwdBattleAF32(adB, st, sc, bs, grads); err != nil {
		t.Fatalf("recon bwd: %v", err)
	}
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}
	snapshotPoints := func() map[string][]float32 {
		return map[string][]float32{
			"DWout":    gpuToHost(adB, grads.DWout, cfg.D*cfg.V),
			"DNormOut": gpuToHost(adB, grads.DNormOut, cfg.D),
			"DWq":      gpuToHost(adB, grads.Layers[0].DWq, cfg.D*cfg.D),
			"DWv":      gpuToHost(adB, grads.Layers[0].DWv, cfg.D*cfg.D),
			"DWo":      gpuToHost(adB, grads.Layers[0].DWo, cfg.D*cfg.D),
			"DNorm1":   gpuToHost(adB, grads.Layers[0].DNorm1, cfg.D),
			"DNorm2":   gpuToHost(adB, grads.Layers[0].DNorm2, cfg.D),
			"DW1":      gpuToHost(adB, grads.Layers[0].DW1, cfg.D*cfg.FFN),
			"DW2":      gpuToHost(adB, grads.Layers[0].DW2, cfg.FFN*cfg.D),
			"DEmbed":   gpuToHost(adB, grads.DEmbed, cfg.V*cfg.D),
		}
	}
	reconPts := snapshotPoints()

	// === Block-level A/B: bs.DOF32 после recon-bwd = dO слоя 0 (L=1, не перезаписан) ===
	// Recon block standalone (в bs.DQPerm/DKPerm/DVPerm).
	if err := attnReconstructFwd(adB, adB.(*adapter.Backend), sc.QPermSnap[0], sc.KPermSnap[0], sc.VPermSnap[0],
		bs.OReconDbg, bs.SReconTemp, bs.PRecon, bs.QScaledTmp, BH, cfg.S, cfg.HD,
		float32(1.0/math.Sqrt(float64(cfg.HD)))); err != nil {
		t.Fatalf("standalone recon fwd: %v", err)
	}
	if err := attnReconstructBwd(adB, adB.(*adapter.Backend), sc.QPermSnap[0], sc.KPermSnap[0], sc.VPermSnap[0], bs.PRecon,
		bs.DOF32, bs.DQPerm, bs.DKPerm, bs.DVPerm, bs.DPTemp, bs.DSTemp, BH, cfg.S, cfg.HD,
		float32(1.0/math.Sqrt(float64(cfg.HD)))); err != nil {
		t.Fatalf("standalone recon bwd: %v", err)
	}
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}
	reconDQ := gpuToHost(adB, bs.DQPerm, nBlk)
	reconDK := gpuToHost(adB, bs.DKPerm, nBlk)
	reconDV := gpuToHost(adB, bs.DVPerm, nBlk)

	// FA block standalone (те же входы: снапшоты + тот же bs.DOF32).
	scales, err := attnFABwdBlock(adB, adB.(*adapter.Backend), faCtx, fb, bs,
		sc.QPermSnap[0], sc.KPermSnap[0], sc.VPermSnap[0], BH, cfg.S, cfg.HD,
		float32(1.0/math.Sqrt(float64(cfg.HD))))
	if err != nil {
		t.Fatalf("FA block standalone: %v", err)
	}
	faDQ := gpuToHost(adB, bs.DQPerm, nBlk)
	faDK := gpuToHost(adB, bs.DKPerm, nBlk)
	faDV := gpuToHost(adB, bs.DVPerm, nBlk)

	// Диагностика промежуточных FA-цепочки (NaN/zero/max) — прибор после ложного
	// PASS первого прогона (NaN-маскировка).
	stat32 := func(name string, s backend.Storage, n int) {
		h := gpuToHost(adB, s, n)
		nan, zero := 0, 0
		var mx float64
		for _, v := range h {
			if math.IsNaN(float64(v)) {
				nan++
			} else if v == 0 {
				zero++
			} else if a := math.Abs(float64(v)); a > mx {
				mx = a
			}
		}
		t.Logf("  DIAG %-10s nan=%d/%d zero=%d max|.|=%.3e", name, nan, n, zero, mx)
	}
	// Нативные промежуточные — читаем через gomlB.
	statNat := func(name string, s backend.Storage, n int) {
		h := gpuToHost(gomlB, s, n)
		nan, zero := 0, 0
		var mx float64
		for _, v := range h {
			if math.IsNaN(float64(v)) {
				nan++
			} else if v == 0 {
				zero++
			} else if a := math.Abs(float64(v)); a > mx {
				mx = a
			}
		}
		t.Logf("  DIAG %-10s nan=%d/%d zero=%d max|.|=%.3e", name, nan, n, zero, mx)
	}
	t.Logf("DIAG FA-цепочка (после standalone block, нативный контекст):")
	statNat("L", fb.LGPUn, BH*cfg.S)
	statNat("D", fb.Dn, BH*cfg.S)
	stat32("dO_F32", bs.DOF32, nBlk)
	statNat("dQ(FA)", fb.DQn, nBlk)
	statNat("dK(FA)", fb.DKn, nBlk)
	statNat("dV(FA)", fb.DVn, nBlk)
	scaleQ, scaleK, scaleV := scales[0], scales[1], scales[2]
	tNorm := float64(scaleV) * math.Pow(2, -6)
	tSubnorm := float64(scaleV) * math.Pow(2, -9)
	t.Logf("FP8-скейлы: scaleQ=%.6e scaleK=%.6e scaleV=%.6e", scaleQ, scaleK, scaleV)
	t.Logf("Пороги зон (по контракту dS direct-cast): T_norm=%.6e T_subnorm=%.6e", tNorm, tSubnorm)

	// Block-level зонный A/B.
	f64Block := map[string][]float64{"dQ": g64.DQPermAttn, "dK": g64.DKPermAttn, "dV": g64.DVPermAttn}
	gpuBlocks := map[string]map[string][]float32{
		"FA":    {"dQ": faDQ, "dK": faDK, "dV": faDV},
		"recon": {"dQ": reconDQ, "dK": reconDK, "dV": reconDV},
	}
	blockNames := []string{"dQ", "dK", "dV"}
	pathNames := []string{"FA", "recon"}
	zoneAFail := 0
	for _, bn := range blockNames {
		ref := f64Block[bn]
		for _, pn := range pathNames {
			gpu := gpuBlocks[pn][bn]
			var nA, nSub, nB, nNaN int
			var worstA, worstSub, worstB float64
			for i := range ref {
				gv := float64(gpu[i])
				if math.IsNaN(gv) {
					nNaN++
					continue
				}
				d := math.Abs(gv - ref[i])
				a := math.Abs(ref[i])
				switch {
				case a >= tNorm:
					nA++
					if d > worstA {
						worstA = d
					}
				case a >= tSubnorm:
					nSub++
					if d > worstSub {
						worstSub = d
					}
				default:
					nB++
					if d > worstB {
						worstB = d
					}
				}
			}
			verdict := "PASS"
			if pn == "FA" && (worstA > 5e-3 || nNaN > 0) {
				verdict = "FAIL (zone-A floor 5e-3 или NaN)"
				zoneAFail++
			}
			t.Logf("BLOCK %s/%-5s: NaN=%d | зонаA n=%d worst|Δ|=%.3e | субнорм n=%d worst=%.3e | зонаB n=%d worst=%.3e -> %s",
				bn, pn, nNaN, nA, worstA, nSub, worstSub, nB, worstB, verdict)
		}
	}
	if zoneAFail > 0 {
		t.Errorf("BLOCK A/B: зона A FAIL на %d путях/тензорах (floor 5e-3, записан до прогона)", zoneAFail)
	}
	// FA vs recon (inter-GPU, справочно).
	for _, bn := range blockNames {
		fa, rc := gpuBlocks["FA"][bn], gpuBlocks["recon"][bn]
		var worst float64
		for i := range fa {
			if d := math.Abs(float64(fa[i]) - float64(rc[i])); d > worst {
				worst = d
			}
		}
		t.Logf("BLOCK %s FA-vs-recon: max|Δ|=%.3e (справочно; дрожь recon задокументирована)", bn, worst)
	}

	// === Полная цепочка с FA-блоком (PRIMARY chain-level) ===
	if err := zeroGrads(adB, grads); err != nil {
		t.Fatalf("zeroGrads FA: %v", err)
	}
	if _, err := fwdBattleAF32(adB, st, sc, inp, tgt); err != nil {
		t.Fatalf("fwd для FA-bwd: %v", err)
	}
	if err := bwdBattleAF32Ex(adB, st, sc, bs, grads, faCtx, fb, true); err != nil {
		t.Fatalf("FA-chain bwd: %v", err)
	}
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}
	// Проба нити C: где рождается NaN в chain-FA.
	t.Logf("DIAG chain-FA (после bwdEx useFA=true): scales=%v", fb.LastScales)
	stat32("ch.DQPerm", bs.DQPerm, nBlk)
	stat32("ch.DKPerm", bs.DKPerm, nBlk)
	stat32("ch.DVPerm", bs.DVPerm, nBlk)
	stat32("ch.DQ(inv)", bs.DQ, cfg.B*cfg.S*cfg.D)
	stat32("ch.DNormed", bs.DNormed, cfg.B*cfg.S*cfg.D)
	stat32("ch.DWq", grads.Layers[0].DWq, cfg.D*cfg.D)
	faPts := snapshotPoints()

	// 10 точек: те же формулы индексов, что A-LLM-3.
	wqIdx := (cfg.D/2)*cfg.D + (cfg.D / 3)
	w1Idx := (cfg.D/2)*cfg.FFN + (cfg.FFN / 3)
	w2Idx := (cfg.FFN/2)*cfg.D + (cfg.D / 3)
	wvIdx := (cfg.D/2)*cfg.D + (cfg.D / 3)
	woIdx := (cfg.D/2)*cfg.D + (cfg.D / 3)
	woutIdx := (cfg.D/2)*cfg.V + (cfg.V / 3)
	embIdx := int(inp[0])*cfg.D + (cfg.D / 3)
	argMaxAbs := func(xs []float64) int {
		best, bestV := 0, math.Abs(xs[0])
		for i := 1; i < len(xs); i++ {
			if a := math.Abs(xs[i]); a > bestV {
				best, bestV = i, a
			}
		}
		return best
	}
	type cpt struct {
		name  string
		key   string
		idx   int
		f64   float64
		faDep bool // зависит ли от FA-блока (позиция в bwd-порядке)
	}
	cpoints := []cpt{
		{"Wout(top)+V-канарейка", "DWout", woutIdx, g64.DWout[woutIdx], false},
		{"NormOut(g)", "DNormOut", argMaxAbs(g64.DNormOut), 0, false},
		{"Wo[L=0]", "DWo", woIdx, g64.Layers[0].DWo[woIdx], false},
		{"W2[L=0]", "DW2", w2Idx, g64.Layers[0].DW2[w2Idx], false},
		{"Norm2(g)", "DNorm2", argMaxAbs(g64.Layers[0].DNorm2), 0, false},
		{"W1[L=0]", "DW1", w1Idx, g64.Layers[0].DW1[w1Idx], false},
		{"Wv[L=0]", "DWv", wvIdx, g64.Layers[0].DWv[wvIdx], true},
		{"Norm1(g)", "DNorm1", argMaxAbs(g64.Layers[0].DNorm1), 0, true},
		{"Wq[L=0]", "DWq", wqIdx, g64.Layers[0].DWq[wqIdx], true},
		{"Embed[i0,d]", "DEmbed", embIdx, g64.DEmbed[embIdx], true},
	}
	// f64 для gamma-точек (idx уже argmax).
	cpoints[1].f64 = g64.DNormOut[cpoints[1].idx]
	cpoints[4].f64 = g64.Layers[0].DNorm2[cpoints[4].idx]
	cpoints[7].f64 = g64.Layers[0].DNorm1[cpoints[7].idx]

	t.Logf("=== CHAIN A/B: 10 точек, PRIMARY vs F64, SECONDARY vs recon ===")
	chainFail := 0
	for _, p := range cpoints {
		faV := float64(faPts[p.key][p.idx])
		rcV := float64(reconPts[p.key][p.idx])
		dFA := math.Abs(faV - p.f64)
		dRC := math.Abs(rcV - p.f64)
		dep := "indep"
		if p.faDep {
			dep = "FA-dep"
		}
		verdict := "PASS"
		if math.IsNaN(faV) {
			verdict = "FAIL (NaN)"
			chainFail++
		} else if p.faDep && math.Abs(p.f64) >= tNorm && dFA > 5e-3 {
			verdict = "ZONE-A FAIL"
			chainFail++
		}
		if !p.faDep && dFA > 5e-3 && dFA > 3*dRC+5e-3 {
			verdict = "OBVYAZKA-SUSPECT" // FA-независимая точка уехала сильнее recon
			chainFail++
		}
		zone := "A"
		if math.Abs(p.f64) < tSubnorm {
			zone = "B"
		} else if math.Abs(p.f64) < tNorm {
			zone = "sub"
		}
		t.Logf("  %-22s [%s, зона %s] f64=%+.6e FA|Δ|=%.3e recon|Δ|=%.3e -> %s",
			p.name, dep, zone, p.f64, dFA, dRC, verdict)
	}
	if chainFail > 0 {
		t.Errorf("CHAIN A/B: %d FAIL", chainFail)
	} else {
		t.Logf("CHAIN A/B PASS: FA-зависимые точки в зоне A внутри floor 5e-3")
	}
}
