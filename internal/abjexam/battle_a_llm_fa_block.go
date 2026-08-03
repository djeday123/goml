package abjexam

// A-LLM-4 Этап 1 (2026-08-03): FA-bwd блок для встройки в эталонную f32recon-обвязку.
//
// Цепочка (ТЗ п.1.1): quantize F32->FP8 -> fa_forward_train (L, O_FP16) ->
// dO cast F32->F16 -> gt_fa_bwd_d_precompute -> gt_fa_bwd_merged ->
// gt_fa_bwd_dk -> gt_fa_bwd_dq -> dQ/dK/dV F32 (real units).
//
// Scale-конвенция (восстановлена из исходников ядер, факт в отчёт):
//   - merged.scale используется ТОЛЬКО для реконструкции P = exp(scale*S - L)
//     (fa_bwd_merged_v1.cu:265-271) => подавать faScale = softmax*scaleQ*scaleK,
//     как в fwd (G1-конвенция).
//   - dV из merged БЕЗ скейла в эпилоге (:485 "no scale") => real units при real dO.
//   - dS квантизуется e4m3 DIRECT CAST без amax-скейла ("ds_gen quant path
//     unchanged") => единицы dS_kernel = dS_real / scaleV (через dP=dO@V_decoded
//     и D=rowsum(O_kernel*dO), O_kernel=O_real/scaleV — согласовано).
//   - dq_new/dk_new умножают аккумулятор на scale в эпилоге (fa_bwd_dq_new.cu:325):
//       dQ_real = (dSnat @ K_d) * scale_dq,  scale_dq = softmax * scaleV * scaleK
//       dK_real = (dST  @ Q_d) * scale_dk,  scale_dk = softmax * scaleV * scaleQ
//
// Порог стирания зоны B (уточнение формулы ТЗ по факту контракта, ДО прогона):
//   стиратель dQ/dK = direct-cast dS в e4m3; в РЕАЛЬНЫХ единицах:
//     T_norm    = scaleV * 2^-6   (наименьшее нормальное e4m3)
//     T_subnorm = scaleV * 2^-9   (субнормальный пол)
//   Для dV стирателя FP8 нет (dO идёт FP16) — зона B не ожидается.
//
// Zero-init контракт (факт "кто требует"): dV/dK/dQ — "must be zero-init"
// (сигнатуры fa_backward.go:112/133/152); OFP16/LGPU перед fa_forward_train
// ([[feedback-fa-buffers-zero-init]]); d_precompute пишет D полностью (не требует).

import (
	"fmt"
	"math"
	"time"

	"github.com/djeday123/goml/backend"
	gomlcuda "github.com/djeday123/goml/backend/cuda"
	gotorchAdapter "github.com/djeday123/goml/backend/gotorch"
)

// faBlockBufs — пре-аллоцированные буферы FA-блока (аллокация ДО первого
// FA-вызова — [[feedback-fa-fwd-scratch-alloc-instability]]).
//
// ДВУХКОНТЕКСТНАЯ схема (находка A-LLM-4): аллокации gotorch-адаптера НЕ валидны
// для ядер libfa_sm120/libfa_bwd_sm120 (bh=4: тихие L=NaN/O=0; bh=128:
// ILLEGAL_ADDRESS). Всё, что трогают FA-ядра, живёт в НАТИВНОМ goml.cuda
// контексте (natB, полученном ДО adapter.Enable); данные ходят host-staging'ом
// (cert-only, ~20MB на вызов).
type faBlockBufs struct {
	adB  backend.Backend // адаптер (квантизация, вход/выход цепочки)
	natB backend.Backend // нативный goml.cuda (все указатели для FA-ядер)
	// Адаптерный контекст (GPU-квантизатор пишет сюда).
	QFP8a, KFP8a, VFP8a    backend.Storage // FP8 [BH, S, HD]
	AmaxQ, AmaxK, AmaxV    backend.Storage // F32 [1]
	ScaleQ, ScaleK, ScaleV backend.Storage // F32 [1]
	// Нативный контекст (входы/выходы FA-ядер).
	QFP8n, KFP8n, VFP8n backend.Storage // FP8 [BH, S, HD]
	OFP16n              backend.Storage // F16 [BH, S, HD]
	LGPUn               backend.Storage // F32 [BH, S]
	DOF16n              backend.Storage // F16 [BH, S, HD]
	Dn                  backend.Storage // F32 [BH, S]
	DSnatN, DSTn        backend.Storage // FP8 padded [BH, S, sPad]
	DVn, DKn, DQn       backend.Storage // F32 [BH, S, HD]
	LastScales          [3]float32      // (scaleQ, scaleK, scaleV) последнего вызова
	// П.4г (A-LLM-6): раздельная атрибуция пятой карты блоков.
	TKernels time.Duration // квант+fa-fwd+D+merged+dk+dq (GPU-стороны)
	TStaging time.Duration // host-staging D2H/H2D (cert-плата)
}

func newFABlockBufs(cfg BattleACfg, adB, natB backend.Backend) (*faBlockBufs, error) {
	BH := cfg.B * cfg.H
	n := BH * cfg.S * cfg.HD
	sPad := (cfg.S + 15) & ^15
	fb := &faBlockBufs{adB: adB, natB: natB}
	var err error
	alA := func(p *backend.Storage, bytes int) bool {
		*p, err = adB.Alloc(bytes)
		return err == nil
	}
	alN := func(p *backend.Storage, bytes int) bool {
		*p, err = natB.Alloc(bytes)
		return err == nil
	}
	ok := alA(&fb.QFP8a, n) && alA(&fb.KFP8a, n) && alA(&fb.VFP8a, n) &&
		alA(&fb.AmaxQ, 4) && alA(&fb.AmaxK, 4) && alA(&fb.AmaxV, 4) &&
		alA(&fb.ScaleQ, 4) && alA(&fb.ScaleK, 4) && alA(&fb.ScaleV, 4) &&
		alN(&fb.QFP8n, n) && alN(&fb.KFP8n, n) && alN(&fb.VFP8n, n) &&
		alN(&fb.OFP16n, n*2) && alN(&fb.LGPUn, BH*cfg.S*4) &&
		alN(&fb.DOF16n, n*2) && alN(&fb.Dn, BH*cfg.S*4) &&
		alN(&fb.DSnatN, BH*cfg.S*sPad) && alN(&fb.DSTn, BH*cfg.S*sPad) &&
		alN(&fb.DVn, n*4) && alN(&fb.DKn, n*4) && alN(&fb.DQn, n*4)
	if !ok {
		return nil, err
	}
	return fb, nil
}

func (fb *faBlockBufs) FreeAll(_ backend.Backend) {
	if fb == nil {
		return
	}
	for _, s := range []backend.Storage{fb.QFP8a, fb.KFP8a, fb.VFP8a,
		fb.AmaxQ, fb.AmaxK, fb.AmaxV, fb.ScaleQ, fb.ScaleK, fb.ScaleV} {
		if s != nil {
			fb.adB.Free(s)
		}
	}
	for _, s := range []backend.Storage{fb.QFP8n, fb.KFP8n, fb.VFP8n, fb.OFP16n,
		fb.LGPUn, fb.DOF16n, fb.Dn, fb.DSnatN, fb.DSTn, fb.DVn, fb.DKn, fb.DQn} {
		if s != nil {
			fb.natB.Free(s)
		}
	}
}

// faRawBytes — сырой device->host download БЕЗ float32-роундтрипа.
// КРИТИЧНО: прокатка произвольных байт через []float32 (Float32frombits ->
// значение -> Float32bits) тихо портит signaling-NaN паттерны (quiet-бит) —
// одна битовая порча на слово. Для FP8/F16 сырья — только байтовый путь.
func faRawBytes(b backend.Backend, s backend.Storage, nBytes int) ([]byte, error) {
	cpuS, err := b.ToDevice(backend.CPU0, s)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, nBytes)
	copy(raw, cpuS.Bytes()[:nBytes])
	return raw, nil
}

// faRawStage — device->host->device перенос сырых байт между аллокаторами.
func faRawStage(srcB backend.Backend, src backend.Storage, dstB backend.Backend,
	dst backend.Storage, bytes int) error {
	raw, err := faRawBytes(srcB, src, bytes)
	if err != nil {
		return err
	}
	_, err = uploadInto(dstB, dst, raw)
	return err
}

// faF32ToF16Host — host-конверсия F32->F16 (для dO staging в нативный контекст).
func faF32ToF16Host(h []float32) []byte {
	out := make([]byte, len(h)*2)
	for i, v := range h {
		b32 := math.Float32bits(v)
		sign := uint16(b32>>16) & 0x8000
		exp := int32(b32>>23)&0xff - 127 + 15
		man := b32 & 0x7fffff
		var h16 uint16
		switch {
		case exp <= 0:
			h16 = sign // FTZ субнормалей F16 (~6e-5 floor, ниже floor 5e-3 зоны A)
		case exp >= 0x1f:
			h16 = sign | 0x7c00
		default:
			h16 = sign | uint16(exp)<<10 | uint16(man>>13)
		}
		out[i*2] = byte(h16)
		out[i*2+1] = byte(h16 >> 8)
	}
	return out
}

// attnFABwdBlock — замена attnReconstructFwd(P)+attnReconstructBwd в bwd:
// вход Qsnap/Ksnap/Vsnap (F32 post-RoPE снапшоты) + bs.DOF32; выход в
// bs.DQPerm/DKPerm/DVPerm (F32, real units, ДО RoPE-bwd — контракт тот же).
// Возвращает скейлы (scaleQ, scaleK, scaleV) для логов/порогов зон.
func attnFABwdBlock(b backend.Backend, gtB *gotorchAdapter.Backend,
	faCtx *gomlcuda.FAContext, fb *faBlockBufs, bs *BattleABwdScratch,
	Qsnap, Ksnap, Vsnap backend.Storage,
	BH, S, HD int, softmaxScale float32) (scales [3]float32, err error) {

	n := BH * S * HD
	natB := fb.natB
	sync := func() {
		if s, ok := b.(interface{ Sync() error }); ok {
			s.Sync()
		}
	}
	syncNat := func() {
		if s, ok := natB.(interface{ Sync() error }); ok {
			s.Sync()
		}
	}
	// (a) Квантизация в адаптерном контексте (GPU-квантизатор боевого fwd):
	// zero amax (atomicMax контракт) + Sync между вызовами.
	zeroF := []byte{0, 0, 0, 0}
	if _, err = uploadInto(b, fb.AmaxQ, zeroF); err != nil {
		return scales, fmt.Errorf("fa-block zero AmaxQ: %w", err)
	}
	if _, err = uploadInto(b, fb.AmaxK, zeroF); err != nil {
		return scales, fmt.Errorf("fa-block zero AmaxK: %w", err)
	}
	if _, err = uploadInto(b, fb.AmaxV, zeroF); err != nil {
		return scales, fmt.Errorf("fa-block zero AmaxV: %w", err)
	}
	// РЕЙС-ФИКС (A-LLM-4): zero-upload amax и quantize-ядро должны быть
	// упорядочены явным Sync — иначе в нагруженном контексте zero-copy
	// исполняется ПОСЛЕ ядра и затирает amax (наблюдалось: amax=0 -> FP8-нули).
	// NB: та же скрытая гонка есть в боевом fwdBattleA (шаг 6) — в отчёт.
	sync()
	// A-LLM-5 квант-контракт O(1): Unit-вариант (scale = amax, decoded <= 1) —
	// контракт FP16-акков v121r (QK/O MMA f16.e4m3.e4m3.f16, П.0a).
	if err = gtB.QuantizeF32ToF8E4M3Unit(Qsnap, fb.QFP8a, fb.ScaleQ, fb.AmaxQ, n); err != nil {
		return scales, fmt.Errorf("fa-block quantize Q: %w", err)
	}
	sync()
	if err = gtB.QuantizeF32ToF8E4M3Unit(Ksnap, fb.KFP8a, fb.ScaleK, fb.AmaxK, n); err != nil {
		return scales, fmt.Errorf("fa-block quantize K: %w", err)
	}
	sync()
	if err = gtB.QuantizeF32ToF8E4M3Unit(Vsnap, fb.VFP8a, fb.ScaleV, fb.AmaxV, n); err != nil {
		return scales, fmt.Errorf("fa-block quantize V: %w", err)
	}
	sync()
	amaxQ := gpuToHost(b, fb.AmaxQ, 1)[0]
	amaxK := gpuToHost(b, fb.AmaxK, 1)[0]
	amaxV := gpuToHost(b, fb.AmaxV, 1)[0]
	// Новая конвенция: scale_X = amax_X (decoded O(1)).
	scaleQ := amaxQ
	scaleK := amaxK
	scaleV := amaxV
	if scaleQ <= 0 {
		scaleQ = 1.0
	}
	if scaleK <= 0 {
		scaleK = 1.0
	}
	if scaleV <= 0 {
		scaleV = 1.0
	}
	scales = [3]float32{scaleQ, scaleK, scaleV}
	faScale := softmaxScale * scaleQ * scaleK

	// (a2) Staging FP8 адаптер -> нативный контекст (host-путь; cert-only).
	tStage := time.Now()
	if err = faRawStage(b, fb.QFP8a, natB, fb.QFP8n, n); err != nil {
		return scales, fmt.Errorf("fa-block stage QFP8: %w", err)
	}
	if err = faRawStage(b, fb.KFP8a, natB, fb.KFP8n, n); err != nil {
		return scales, fmt.Errorf("fa-block stage KFP8: %w", err)
	}
	if err = faRawStage(b, fb.VFP8a, natB, fb.VFP8n, n); err != nil {
		return scales, fmt.Errorf("fa-block stage VFP8: %w", err)
	}
	fb.TStaging += time.Since(tStage)

	// (b) fa_forward_train (нативный контекст) -> O_FP16 + L. Zero-init ДО вызова.
	zeroNat := func(s backend.Storage, bytes int) error {
		z := make([]byte, bytes)
		_, e := uploadInto(natB, s, z)
		return e
	}
	if err = zeroNat(fb.OFP16n, n*2); err != nil {
		return scales, fmt.Errorf("fa-block zero OFP16: %w", err)
	}
	if err = zeroNat(fb.LGPUn, BH*S*4); err != nil {
		return scales, fmt.Errorf("fa-block zero LGPU: %w", err)
	}
	tKern := time.Now()
	if err = faCtx.ForwardTrain(devPtr(fb.QFP8n), devPtr(fb.KFP8n), devPtr(fb.VFP8n),
		devPtr(fb.OFP16n), devPtr(fb.LGPUn),
		int32(BH), int32(S), int32(HD), 0, 0, faScale, 0); err != nil {
		return scales, fmt.Errorf("fa-block fa_forward_train: %w", err)
	}
	syncNat()
	fb.TKernels += time.Since(tKern)

	// (c) dO: адаптерный DOF32 -> host F16 -> нативный DOF16n.
	tStage = time.Now()
	doHost := gpuToHost(b, bs.DOF32, n)
	if _, err = uploadInto(natB, fb.DOF16n, faF32ToF16Host(doHost)); err != nil {
		return scales, fmt.Errorf("fa-block dO stage: %w", err)
	}
	fb.TStaging += time.Since(tStage)

	// (d) D-precompute (D пишется полностью, zero-init не требует).
	tKern = time.Now()
	if err = gomlcuda.FABwdDPrecompute(devPtr(fb.OFP16n), devPtr(fb.DOF16n), devPtr(fb.Dn),
		BH, S, HD, 0); err != nil {
		return scales, fmt.Errorf("fa-block d_precompute: %w", err)
	}
	syncNat()

	// (e) Zero-init dV/dK/dQ (контракт "must be zero-init", 3/4 ядер).
	if err = zeroNat(fb.DVn, n*4); err != nil {
		return scales, fmt.Errorf("fa-block zero dV: %w", err)
	}
	if err = zeroNat(fb.DKn, n*4); err != nil {
		return scales, fmt.Errorf("fa-block zero dK: %w", err)
	}
	if err = zeroNat(fb.DQn, n*4); err != nil {
		return scales, fmt.Errorf("fa-block zero dQ: %w", err)
	}

	// (f) merged -> dk_new -> dq_new. Причинность: causal=0, window=0 (П-5).
	if err = gomlcuda.FABwdMerged(devPtr(fb.QFP8n), devPtr(fb.KFP8n), devPtr(fb.VFP8n),
		devPtr(fb.DOF16n), devPtr(fb.LGPUn), devPtr(fb.Dn),
		devPtr(fb.DSnatN), devPtr(fb.DSTn), devPtr(fb.DVn),
		BH, S, HD, 0, 0, faScale, 0); err != nil {
		return scales, fmt.Errorf("fa-block merged: %w", err)
	}
	syncNat()
	scaleDK := softmaxScale * scaleV * scaleQ
	scaleDQ := softmaxScale * scaleV * scaleK
	if err = gomlcuda.FABwdDK(devPtr(fb.QFP8n), devPtr(fb.DSTn), devPtr(fb.DKn),
		BH, S, HD, 0, 0, scaleDK, 0); err != nil {
		return scales, fmt.Errorf("fa-block dk_new: %w", err)
	}
	if err = gomlcuda.FABwdDQ(devPtr(fb.KFP8n), devPtr(fb.DSnatN), devPtr(fb.DQn),
		BH, S, HD, 0, 0, scaleDQ, 0); err != nil {
		return scales, fmt.Errorf("fa-block dq_new: %w", err)
	}
	syncNat()
	fb.TKernels += time.Since(tKern)

	// (g) Выходы: нативные F32 -> host -> адаптерные контрактные буферы цепочки.
	tStage = time.Now()
	if err = faRawStage(natB, fb.DQn, b, bs.DQPerm, n*4); err != nil {
		return scales, fmt.Errorf("fa-block stage dQ out: %w", err)
	}
	if err = faRawStage(natB, fb.DKn, b, bs.DKPerm, n*4); err != nil {
		return scales, fmt.Errorf("fa-block stage dK out: %w", err)
	}
	if err = faRawStage(natB, fb.DVn, b, bs.DVPerm, n*4); err != nil {
		return scales, fmt.Errorf("fa-block stage dV out: %w", err)
	}
	sync()
	fb.TStaging += time.Since(tStage)
	// Верификация staging-канала выходов (bit-exact nat -> adapter): дешёвая
	// плата за пойманные в этой сессии SNaN-порчу и молчаливые не-доезды.
	natRaw, e1 := faRawBytes(natB, fb.DQn, n*4)
	adRaw, e2 := faRawBytes(b, bs.DQPerm, n*4)
	if e1 == nil && e2 == nil {
		for i := range natRaw {
			if natRaw[i] != adRaw[i] {
				return scales, fmt.Errorf("fa-block STAGING VERIFY FAIL: dQ byte %d nat=%02x ad=%02x", i, natRaw[i], adRaw[i])
			}
		}
	}
	return scales, nil
}
