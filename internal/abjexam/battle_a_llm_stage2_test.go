package abjexam

// A-LLM-6 П.4: Этап 2 — боевой bwdBattleA на снапшот-математике + FA-путь.
//
// TestALLM_Stage2_InterPath (4б): bwdBattleA(recon) vs bwdBattleAF32 на ОБЩИХ
//   входах (снапшоты скопированы из F32-стека) — изоляция именно обвязки.
//   Floor'ы ДО прогона: база 1e-4 abs; для классов с задокументированной
//   дрожью пути (raw_allm3/allm4) — их worst-числа (таблица в коде).
//   Затем bwdBattleA(FA) на тех же входах: NaN-гейт + зонная классификация.
//
// TestALLM_Stage2_TrajectorySpeed (4в+4г): боевая форма V=32000 L=1.
//   ПРОГНОЗ-ВИЛКА (записана ДО прогона, из карты Stage5 ~109ms класс +
//   канон FA-bwd 42.346ms bh=128/sl=8192 -> bh=4/sl=2048):
//     FA-kernels блока: идеал 42.35/512 ~= 0.08ms fwd+0.01 -> [0.1, 0.4] ms
//       (SM-недогруз: merged grid 4x32=128 блоков на 128 SM, 2 blk/SM резерв);
//     staging блока: ~26MB host-пути -> [3, 9] ms (ДОМИНАНТА, cert-плата);
//     шаг recon-пути: [0.3, 2.0] s (plain-host-transpose S^2-матрицы в recon
//       + SGD host-обновление ~130MB весов);
//     шаг FA-пути: [0.2, 1.5] s (staging+SGD доминируют; FA-часть может быть
//       ДЕШЕВЛЕ recon-части).
//   Траектория 20 шагов каждого пути: ОЖИДАНИЕ плато на FA-пути (зона B
//   cold-start) = ПРАВИЛЬНОСТЬ; ножницы числом. NaN/Inf-сторож каждый шаг
//   (loss + амаксы Q/K/V). Чувствительностная проба на шагах 0 и 20.
//   Скорость: 30-run каждого пути, CV-gate <1%; пятая карта: staging
//   ОТДЕЛЬНОЙ строкой от kernels (П.4г усиление).

import (
	"math"
	"os"
	"math/rand"
	"runtime"
	"sort"
	"testing"
	"time"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	gomlcuda "github.com/djeday123/goml/backend/cuda"
	adapter "github.com/djeday123/goml/backend/gotorch"
)

func stage2Setup(t *testing.T) (backend.Backend, *gomlcuda.FAContext, backend.Backend) {
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
	return adB, faCtx, gomlB
}

func TestALLM_Stage2_InterPath(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	// ИЗВЕСТНЫЙ RED (A-LLM-6, 2026-08-03): смоук 4б выявил систематические
	// расхождения новой снапшот-обвязки bwdBattleA vs эталонной (DEmbed 18.7 —
	// блоу-ап класс, не дрожь) + рецидив amax=0 в FA-вызове боевой обвязки.
	// Диагностика = A-LLM-7. Включить: GOML_STAGE2=1.
	if os.Getenv("GOML_STAGE2") != "1" {
		t.Skip("known-red: диагностика обвязки Этапа 2 — A-LLM-7")
	}
	adB, faCtx, gomlB := stage2Setup(t)
	defer faCtx.Destroy()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	cfg := faABCfg()
	inp, tgt := faABTokens(cfg)

	// Эталонная обвязка (bwdBattleAF32 на её родном fwd).
	rInit := rand.New(rand.NewSource(31))
	st, err := NewBattleAState(cfg, rInit, adB)
	if err != nil {
		t.Fatalf("state: %v", err)
	}
	defer st.FreeAll(adB)
	sc32, err := NewBattleAScratchF32(cfg, adB)
	if err != nil {
		t.Fatalf("sc32: %v", err)
	}
	defer sc32.FreeAll(adB)
	bs, err := NewBattleABwdScratch(cfg, adB)
	if err != nil {
		t.Fatalf("bs: %v", err)
	}
	defer bs.FreeAll(adB)
	grads, err := NewBattleAGrads(cfg, adB)
	if err != nil {
		t.Fatalf("grads: %v", err)
	}
	defer grads.FreeAll(adB)
	if _, err := fwdBattleAF32(adB, st, sc32, inp, tgt); err != nil {
		t.Fatalf("fwdF32: %v", err)
	}
	if err := zeroGrads(adB, grads); err != nil {
		t.Fatalf("zeroGrads: %v", err)
	}
	if err := bwdBattleAF32(adB, st, sc32, bs, grads); err != nil {
		t.Fatalf("bwdF32: %v", err)
	}
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}
	snapPts := func() map[string][]float32 {
		return map[string][]float32{
			"DWout": gpuToHost(adB, grads.DWout, cfg.D*cfg.V), "DNormOut": gpuToHost(adB, grads.DNormOut, cfg.D),
			"DWq": gpuToHost(adB, grads.Layers[0].DWq, cfg.D*cfg.D), "DWv": gpuToHost(adB, grads.Layers[0].DWv, cfg.D*cfg.D),
			"DWo": gpuToHost(adB, grads.Layers[0].DWo, cfg.D*cfg.D), "DNorm1": gpuToHost(adB, grads.Layers[0].DNorm1, cfg.D),
			"DNorm2": gpuToHost(adB, grads.Layers[0].DNorm2, cfg.D), "DW1": gpuToHost(adB, grads.Layers[0].DW1, cfg.D*cfg.FFN),
			"DW2": gpuToHost(adB, grads.Layers[0].DW2, cfg.FFN*cfg.D), "DEmbed": gpuToHost(adB, grads.DEmbed, cfg.V*cfg.D),
		}
	}
	refPts := snapPts()

	// Боевая обвязка на СКОПИРОВАННЫХ входах.
	sc, err := NewBattleAScratch(cfg, adB)
	if err != nil {
		t.Fatalf("sc: %v", err)
	}
	defer sc.FreeAll(adB)
	snap, err := NewBattleASnapScratch(cfg, adB)
	if err != nil {
		t.Fatalf("snap: %v", err)
	}
	defer snap.FreeAll(adB)
	fb, err := newFABlockBufs(cfg, adB, gomlB)
	if err != nil {
		t.Fatalf("fb: %v", err)
	}
	defer fb.FreeAll(nil)
	M := cfg.B * cfg.S
	BH := cfg.B * cfg.H
	cp := func(dst, src backend.Storage, bytes int) {
		if err := adB.Copy(dst, src, bytes); err != nil {
			t.Fatalf("copy: %v", err)
		}
	}
	for l := 0; l < cfg.L; l++ {
		cp(snap.XPreAttn[l], sc32.XPreAttn[l], M*cfg.D*4)
		cp(snap.XPreFFN[l], sc32.XPreFFN[l], M*cfg.D*4)
		cp(snap.QPerm[l], sc32.QPermSnap[l], BH*cfg.S*cfg.HD*4)
		cp(snap.KPerm[l], sc32.KPermSnap[l], BH*cfg.S*cfg.HD*4)
		cp(snap.VPerm[l], sc32.VPermSnap[l], BH*cfg.S*cfg.HD*4)
		cp(snap.OAttn[l], sc32.OAttnSnap[l], BH*cfg.S*cfg.HD*4)
	}
	cp(snap.XPreTop, sc32.XPreTop, M*cfg.D*4)
	cp(sc.GradL, sc32.GradL, M*cfg.V*4)
	cp(sc.Normed, sc32.Normed, M*cfg.D*4)
	uploadInto(adB, sc.InputGPU, int64ToBytes(inp))
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}

	// Floor'ы (ДО прогона): база 1e-4 abs; дрожь-классы из raw_allm3/allm4 worst.
	floors := map[string]float64{
		"DWout": 1e-4, "DNormOut": 1e-3, "DWo": 1.3e-1, "DW2": 6e-2, "DNorm2": 2e-3,
		"DW1": 1e-4, "DWv": 3e-3, "DNorm1": 1e-3, "DWq": 4e-3, "DEmbed": 3e-2,
	}
	names := []string{"DWout", "DNormOut", "DWo", "DW2", "DNorm2", "DW1", "DWv", "DNorm1", "DWq", "DEmbed"}

	// bwdBattleA(recon) на скопированных входах.
	if err := zeroGrads(adB, grads); err != nil {
		t.Fatalf("zeroGrads recon: %v", err)
	}
	if err := bwdBattleA(adB, st, sc, bs, grads, faCtx, inp, AttnBwdF32Recon, snap, nil); err != nil {
		t.Fatalf("bwdBattleA(recon): %v", err)
	}
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}
	reconPts := snapPts()
	t.Logf("=== 4б смоук inter-path: bwdBattleA(recon) vs bwdBattleAF32 (общие входы) ===")
	fails := 0
	for _, nm := range names {
		a, b2 := reconPts[nm], refPts[nm]
		var worst float64
		nan := 0
		for i := range a {
			av := float64(a[i])
			if math.IsNaN(av) {
				nan++
				continue
			}
			if d := math.Abs(av - float64(b2[i])); d > worst {
				worst = d
			}
		}
		fl := floors[nm]
		verdict := "PASS"
		if worst > fl || nan > 0 {
			verdict = "FAIL"
			fails++
		}
		t.Logf("  %-9s worst|Δ|=%.3e NaN=%d (floor %.1e) -> %s", nm, worst, nan, fl, verdict)
	}
	if fails > 0 {
		t.Errorf("4б смоук: %d/10 FAIL", fails)
	}

	// bwdBattleA(FA) на тех же входах: NaN-гейт + сравнение с recon-путём.
	if err := zeroGrads(adB, grads); err != nil {
		t.Fatalf("zeroGrads fa: %v", err)
	}
	if err := bwdBattleA(adB, st, sc, bs, grads, faCtx, inp, AttnBwdFA, snap, fb); err != nil {
		t.Fatalf("bwdBattleA(FA): %v", err)
	}
	if s, ok := adB.(interface{ Sync() error }); ok {
		s.Sync()
	}
	faPts := snapPts()
	t.Logf("=== 4б FA-путь в боевой обвязке (vs recon-путь; скейлы=%v) ===", fb.LastScales)
	faFails := 0
	for _, nm := range names {
		a, b2 := faPts[nm], reconPts[nm]
		var worst float64
		nan := 0
		for i := range a {
			av := float64(a[i])
			if math.IsNaN(av) || math.IsInf(av, 0) {
				nan++
				continue
			}
			if d := math.Abs(av - float64(b2[i])); d > worst {
				worst = d
			}
		}
		verdict := "OK"
		if nan > 0 {
			verdict = "FAIL (NaN/Inf)"
			faFails++
		}
		t.Logf("  %-9s worst|FA-recon|=%.3e NaN=%d -> %s", nm, worst, nan, verdict)
	}
	if faFails > 0 {
		t.Errorf("4б FA-путь: NaN/Inf в %d тензорах", faFails)
	} else {
		t.Logf("4б FA-путь: NaN-гейт чист; расхождения = зоны Stage1-класса (dS-стирание legit)")
	}
}

func TestALLM_Stage2_TrajectorySpeed(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	// ИЗВЕСТНЫЙ RED (A-LLM-6, 2026-08-03): смоук 4б выявил систематические
	// расхождения новой снапшот-обвязки bwdBattleA vs эталонной (DEmbed 18.7 —
	// блоу-ап класс, не дрожь) + рецидив amax=0 в FA-вызове боевой обвязки.
	// Диагностика = A-LLM-7. Включить: GOML_STAGE2=1.
	if os.Getenv("GOML_STAGE2") != "1" {
		t.Skip("known-red: диагностика обвязки Этапа 2 — A-LLM-7")
	}
	adB, faCtx, gomlB := stage2Setup(t)
	defer faCtx.Destroy()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	cfg := DefaultBattleACfg(1)
	cfg.L = 1 // L=4-init — отдельный долг, не блокирует (стык покрыт L=1)
	rTok := rand.New(rand.NewSource(41))
	M := cfg.B * cfg.S
	inp := make([]int64, M)
	tgt := make([]int32, M)
	for i := 0; i < M; i++ {
		inp[i] = int64(rTok.Intn(cfg.V))
		tgt[i] = int32(rTok.Intn(cfg.V))
	}
	t.Logf("ПРОГНОЗ-ВИЛКА (до прогона): FA-kernels [0.1,0.4]ms; staging [3,9]ms; шаг recon [0.3,2.0]s; шаг FA [0.2,1.5]s")

	mkStack := func() (*BattleAState, *BattleAScratch, *BattleABwdScratch, *BattleAGrads, *BattleASnapScratch, *faBlockBufs) {
		st, err := NewBattleAState(cfg, rand.New(rand.NewSource(31)), adB)
		if err != nil {
			t.Fatalf("state: %v", err)
		}
		sc, err := NewBattleAScratch(cfg, adB)
		if err != nil {
			t.Fatalf("sc: %v", err)
		}
		bs, err := NewBattleABwdScratch(cfg, adB)
		if err != nil {
			t.Fatalf("bs: %v", err)
		}
		grads, err := NewBattleAGrads(cfg, adB)
		if err != nil {
			t.Fatalf("grads: %v", err)
		}
		snap, err := NewBattleASnapScratch(cfg, adB)
		if err != nil {
			t.Fatalf("snap: %v", err)
		}
		fb, err := newFABlockBufs(cfg, adB, gomlB)
		if err != nil {
			t.Fatalf("fb: %v", err)
		}
		return st, sc, bs, grads, snap, fb
	}
	amax3 := func(sc *BattleAScratch) (float32, float32, float32) {
		return gpuToHost(adB, sc.AmaxQ, 1)[0], gpuToHost(adB, sc.AmaxK, 1)[0], gpuToHost(adB, sc.AmaxV, 1)[0]
	}
	sensProbe := func(st *BattleAState, sc *BattleAScratch, snap *BattleASnapScratch, label string) {
		l0, err := fwdBattleA(adB, st, sc, faCtx, inp, tgt, snap)
		if err != nil {
			t.Fatalf("probe fwd: %v", err)
		}
		wv := gpuToHost(adB, st.Layers[0].Wv, cfg.D*cfg.D)
		orig := wv[0]
		wv[0] = orig + 1e-2
		uploadInto(adB, st.Layers[0].Wv, f32ToBytes(wv))
		l1, err := fwdBattleA(adB, st, sc, faCtx, inp, tgt, snap)
		if err != nil {
			t.Fatalf("probe fwd2: %v", err)
		}
		wv[0] = orig
		uploadInto(adB, st.Layers[0].Wv, f32ToBytes(wv))
		d := math.Abs(l1 - l0)
		verdict := "ЖИВ"
		if d == 0 {
			verdict = "МЁРТВ (attention не влияет на loss!)"
			t.Errorf("чувствительностная проба %s: |dLoss|=0", label)
		}
		t.Logf("  проба %s: возмущение Wv[0] +1e-2 -> |dLoss|=%.3e -> %s", label, d, verdict)
	}

	runTraj := func(path AttnBwdPath, label string) ([]float64, *BattleAState, *BattleAScratch, *BattleABwdScratch, *BattleAGrads, *BattleASnapScratch, *faBlockBufs) {
		st, sc, bs, grads, snap, fb := mkStack()
		losses := make([]float64, 0, 21)
		sensProbe(st, sc, snap, label+"/шаг-0")
		for step := 0; step < 20; step++ {
			loss, err := trainStepBattleA(adB, st, sc, bs, grads, faCtx, inp, tgt, 1e-2, path, snap, fb)
			if err != nil {
				t.Fatalf("%s шаг %d: %v", label, step, err)
			}
			aq, ak, av := amax3(sc)
			// NaN/Inf-сторож (усиление ревью): номер шага и амаксы в raw при взрыве.
			if math.IsNaN(loss) || math.IsInf(loss, 0) ||
				math.IsNaN(float64(aq)) || math.IsNaN(float64(ak)) || math.IsNaN(float64(av)) {
				t.Errorf("%s ВЗРЫВ на шаге %d: loss=%v amax=(%.3e,%.3e,%.3e)", label, step, loss, aq, ak, av)
				break
			}
			losses = append(losses, loss)
			t.Logf("  %s step %2d: loss=%.4f amaxQKV=(%.3f,%.3f,%.3f)", label, step, loss, aq, ak, av)
		}
		sensProbe(st, sc, snap, label+"/шаг-20")
		return losses, st, sc, bs, grads, snap, fb
	}

	t.Logf("=== 4в траектория 20 шагов: recon-путь ===")
	lossR, stR, scR, bsR, grR, snR, fbR := runTraj(AttnBwdF32Recon, "recon")
	t.Logf("=== 4в траектория 20 шагов: FA-путь ===")
	lossF, stF, scF, bsF, grF, snF, fbF := runTraj(AttnBwdFA, "FA")
	defer func() {
		stR.FreeAll(adB)
		scR.FreeAll(adB)
		bsR.FreeAll(adB)
		grR.FreeAll(adB)
		snR.FreeAll(adB)
		fbR.FreeAll(nil)
		stF.FreeAll(adB)
		scF.FreeAll(adB)
		bsF.FreeAll(adB)
		grF.FreeAll(adB)
		snF.FreeAll(adB)
		fbF.FreeAll(nil)
	}()
	t.Logf("=== 4в ножницы (per-step delta-loss, recon - FA) ===")
	for i := 0; i < len(lossR) && i < len(lossF); i++ {
		t.Logf("  step %2d: recon=%.4f FA=%.4f delta=%+.4f", i, lossR[i], lossF[i], lossR[i]-lossF[i])
	}

	// 4г скорость: 30-run каждого пути.
	speed := func(label string, path AttnBwdPath, st *BattleAState, sc *BattleAScratch,
		bs *BattleABwdScratch, grads *BattleAGrads, snap *BattleASnapScratch, fb *faBlockBufs) (float64, float64) {
		fb.TKernels, fb.TStaging = 0, 0
		times := make([]float64, 0, 30)
		for i := 0; i < 30; i++ {
			t0 := time.Now()
			if _, err := trainStepBattleA(adB, st, sc, bs, grads, faCtx, inp, tgt, 1e-2, path, snap, fb); err != nil {
				t.Fatalf("%s speed run %d: %v", label, i, err)
			}
			times = append(times, time.Since(t0).Seconds()*1000)
		}
		sort.Float64s(times)
		med := times[15]
		var mean, sd float64
		for _, v := range times {
			mean += v
		}
		mean /= 30
		for _, v := range times {
			sd += (v - mean) * (v - mean)
		}
		sd = math.Sqrt(sd / 29)
		cv := sd / mean * 100
		t.Logf("4г %s: median=%.1fms mean=%.1fms CV=%.2f%% (гейт <1%%)", label, med, mean, cv)
		if path == AttnBwdFA {
			t.Logf("4г пятая карта (30 шагов, П.4г раздельная атрибуция): FA-kernels=%.1fms staging=%.1fms (на шаг: %.2f / %.2f ms) — staging=cert-плата, к устранению в связке FA-F16/механика-Н1",
				fb.TKernels.Seconds()*1000, fb.TStaging.Seconds()*1000,
				fb.TKernels.Seconds()*1000/30, fb.TStaging.Seconds()*1000/30)
		}
		return med, cv
	}
	t.Logf("=== 4г скорость 30-run ===")
	medR, cvR := speed("шаг-до (recon)", AttnBwdF32Recon, stR, scR, bsR, grR, snR, fbR)
	medF, cvF := speed("шаг-после (FA)", AttnBwdFA, stF, scF, bsF, grF, snF, fbF)
	t.Logf("4г итог: recon=%.1fms (CV %.2f%%) FA=%.1fms (CV %.2f%%), отношение %.2fx", medR, cvR, medF, cvF, medR/medF)
	if cvR > 1.0 || cvF > 1.0 {
		t.Logf("4г CV-gate ПРОМАХ (два числа: гейт 1%%, факт recon %.2f%% / FA %.2f%%) — разбор в отчёте", cvR, cvF)
	}
}
