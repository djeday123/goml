package abjexam

// A-LLM-1 Stage 4 (2026-07-26): F32-reconstruct attention forward + backward.
// Reference-quality path: НЕ используется в hot-path bwd (там либо FA-bwd, либо reconstruct-loop).
// Единственное назначение — grad-consistency СЕРТИФИКАТ и amax-verify.
//
// Math (row-wise, batch-per-head [S, HD]):
//   Fwd: S = Q @ K^T * scale;  P = softmax(S);  O = P @ V.
//   Bwd (given dO):
//     dV = P^T @ dO
//     dP = dO @ V^T
//     dS = P * (dP - sum_k(P * dP))   -- our softmax_bwd_f32 PTX
//     dQ = dS @ K * scale
//     dK = dS^T @ Q * scale
//
// Layout: [BH, S, HD] row-major. Loop per-batch via MatMulF32Ex (row-major, trans-flags).

import (
	"fmt"
	"unsafe"

	"github.com/djeday123/goml/backend"
	gotorchAdapter "github.com/djeday123/goml/backend/gotorch"
	"github.com/djeday123/goml/core"
)

// launchSoftmaxBwd -- вызов softmax_bwd_f32 PTX kernel.
// P, dP, dS -- device ptrs. P/dP inputs, dS output. rows = BH*S, cols = S.
func launchSoftmaxBwd(b backend.Backend, PPtr, dPPtr, dSPtr uintptr, rows, cols int) error {
	l, ok := b.(interface {
		Launch(name string, gx, gy, gz, bx, by, bz uint32, params []unsafe.Pointer) error
	})
	if !ok {
		return fmt.Errorf("backend has no Launch")
	}
	ru := uint32(rows)
	cu := uint32(cols)
	params := []unsafe.Pointer{
		unsafe.Pointer(&PPtr), unsafe.Pointer(&dPPtr), unsafe.Pointer(&dSPtr),
		unsafe.Pointer(&ru), unsafe.Pointer(&cu),
	}
	return l.Launch("softmax_bwd_f32", uint32(rows), 1, 1, 1, 1, 1, params)
}

// attnReconstructFwd -- F32 reference forward.
//   Q, K, V: [BH, S, HD] F32.
//   O: [BH, S, HD] F32 (out).
//   Stemp: [BH, S, S] F32 (scratch for QK^T*scale, потом souverwritten softmax(S)).
//   Pout: [BH, S, S] F32 (out — сохраняем P для использования в bwd).
//   Qscaled: [BH, S, HD] F32 (scratch — Q * scale, чтобы избежать D2H/H2D на S).
//   scale = softmax_scale (1/sqrt(hd)).
func attnReconstructFwd(b backend.Backend, gtB *gotorchAdapter.Backend,
	Q, K, V, O, Stemp, Pout, Qscaled backend.Storage,
	BH, S, HD int, scale float32) error {
	// Pre-scale Q on host once per fwd (Qscaled = Q * scale), then
	// S = Qscaled @ K^T = scale * (Q @ K^T) -- same result, no per-batch D2H.
	nQ := BH * S * HD
	hostQ := gpuToHost(b, Q, nQ)
	scaledQ := make([]float32, nQ)
	for i := 0; i < nQ; i++ {
		scaledQ[i] = hostQ[i] * scale
	}
	if _, err := uploadInto(b, Qscaled, f32ToBytes(scaledQ)); err != nil {
		return fmt.Errorf("attn recon fwd: upload Qscaled: %w", err)
	}

	if BH == 1 {
		// Fast path (certificate form): use whole tensors, no sliceStore.
		if err := gtB.MatMulF32Ex(Qscaled, K, Stemp, S, S, HD, false, true); err != nil {
			return fmt.Errorf("attn recon fwd: S=QK^T: %w", err)
		}
		if err := b.Softmax(Pout, Stemp, core.Shape{S, S}, -1, core.Float32); err != nil {
			return fmt.Errorf("attn recon fwd: softmax: %w", err)
		}
		if err := gtB.MatMulF32Ex(Pout, V, O, S, HD, S, false, false); err != nil {
			return fmt.Errorf("attn recon fwd: O=P@V: %w", err)
		}
		return nil
	}
	qsBase := devPtr(Qscaled)
	kBase := devPtr(K)
	vBase := devPtr(V)
	oBase := devPtr(O)
	sBase := devPtr(Stemp)
	pBase := devPtr(Pout)
	qkvStride := uintptr(S * HD * 4)
	ssStride := uintptr(S * S * 4)
	for bi := 0; bi < BH; bi++ {
		qsb := &sliceStore{ptr: qsBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
		kb := &sliceStore{ptr: kBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
		vb := &sliceStore{ptr: vBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
		ob := &sliceStore{ptr: oBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
		sb := &sliceStore{ptr: sBase + uintptr(bi)*ssStride, byteLen: S * S * 4}
		pb := &sliceStore{ptr: pBase + uintptr(bi)*ssStride, byteLen: S * S * 4}

		if err := gtB.MatMulF32Ex(qsb, kb, sb, S, S, HD, false, true); err != nil {
			return fmt.Errorf("attn recon fwd batch %d: S=QK^T: %w", bi, err)
		}
		if err := b.Softmax(pb, sb, core.Shape{S, S}, -1, core.Float32); err != nil {
			return fmt.Errorf("attn recon fwd batch %d: softmax: %w", bi, err)
		}
		if err := gtB.MatMulF32Ex(pb, vb, ob, S, HD, S, false, false); err != nil {
			return fmt.Errorf("attn recon fwd batch %d: O=P@V: %w", bi, err)
		}
	}
	return nil
}

// attnReconstructBwd -- F32 reference backward.
//   Q, K, V: [BH, S, HD] F32 (inputs, unchanged).
//   P: [BH, S, S] F32 (from fwd).
//   dO: [BH, S, HD] F32 (input grad).
//   dQ, dK, dV: [BH, S, HD] F32 (out).
//   dPtemp, dStemp: [BH, S, S] F32 (scratch).
//   scale = softmax_scale.
func attnReconstructBwd(b backend.Backend, gtB *gotorchAdapter.Backend,
	Q, K, V, P, dO, dQ, dK, dV, dPtemp, dStemp backend.Storage,
	BH, S, HD int, scale float32) error {
	if BH == 1 {
		// Certificate path: whole tensors, no sliceStore.
		if err := gtB.MatMulF32Ex(P, dO, dV, S, HD, S, true, false); err != nil {
			return fmt.Errorf("attn recon bwd: dV=P^T@dO: %w", err)
		}
		if err := gtB.MatMulF32Ex(dO, V, dPtemp, S, S, HD, false, true); err != nil {
			return fmt.Errorf("attn recon bwd: dP=dO@V^T: %w", err)
		}
		if err := launchSoftmaxBwd(b, devPtr(P), devPtr(dPtemp), devPtr(dStemp), S, S); err != nil {
			return fmt.Errorf("attn recon bwd: softmax_bwd_f32: %w", err)
		}
		if err := gtB.MatMulF32Ex(dStemp, K, dQ, S, HD, S, false, false); err != nil {
			return fmt.Errorf("attn recon bwd: dQ=dS@K: %w", err)
		}
		if err := gtB.MatMulF32Ex(dStemp, Q, dK, S, HD, S, true, false); err != nil {
			return fmt.Errorf("attn recon bwd: dK=dS^T@Q: %w", err)
		}
	} else {
		qBase := devPtr(Q)
		kBase := devPtr(K)
		vBase := devPtr(V)
		pBase := devPtr(P)
		doBase := devPtr(dO)
		dqBase := devPtr(dQ)
		dkBase := devPtr(dK)
		dvBase := devPtr(dV)
		dpBase := devPtr(dPtemp)
		dsBase := devPtr(dStemp)
		qkvStride := uintptr(S * HD * 4)
		ssStride := uintptr(S * S * 4)

		for bi := 0; bi < BH; bi++ {
			vb := &sliceStore{ptr: vBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
			pb := &sliceStore{ptr: pBase + uintptr(bi)*ssStride, byteLen: S * S * 4}
			dob := &sliceStore{ptr: doBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
			dvb := &sliceStore{ptr: dvBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
			dpb := &sliceStore{ptr: dpBase + uintptr(bi)*ssStride, byteLen: S * S * 4}

			if err := gtB.MatMulF32Ex(pb, dob, dvb, S, HD, S, true, false); err != nil {
				return fmt.Errorf("attn recon bwd batch %d: dV=P^T@dO: %w", bi, err)
			}
			if err := gtB.MatMulF32Ex(dob, vb, dpb, S, S, HD, false, true); err != nil {
				return fmt.Errorf("attn recon bwd batch %d: dP=dO@V^T: %w", bi, err)
			}
		}
		if err := launchSoftmaxBwd(b, pBase, dpBase, dsBase, BH*S, S); err != nil {
			return fmt.Errorf("attn recon bwd: softmax_bwd_f32: %w", err)
		}
		for bi := 0; bi < BH; bi++ {
			qb := &sliceStore{ptr: qBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
			kb := &sliceStore{ptr: kBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
			dqb := &sliceStore{ptr: dqBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
			dkb := &sliceStore{ptr: dkBase + uintptr(bi)*qkvStride, byteLen: S * HD * 4}
			dsb := &sliceStore{ptr: dsBase + uintptr(bi)*ssStride, byteLen: S * S * 4}

			if err := gtB.MatMulF32Ex(dsb, kb, dqb, S, HD, S, false, false); err != nil {
				return fmt.Errorf("attn recon bwd batch %d: dQ=dS@K: %w", bi, err)
			}
			if err := gtB.MatMulF32Ex(dsb, qb, dkb, S, HD, S, true, false); err != nil {
				return fmt.Errorf("attn recon bwd batch %d: dK=dS^T@Q: %w", bi, err)
			}
		}
	}
	// Post-scale dQ, dK: (chain rule through S = Q@K^T * scale).
	// Simplest: host D2H * H2D. Small forms — trivial. Formalise later через kernel.
	nElem := BH * S * HD
	hostDQ := gpuToHost(b, dQ, nElem)
	hostDK := gpuToHost(b, dK, nElem)
	for i := range hostDQ {
		hostDQ[i] *= scale
	}
	for i := range hostDK {
		hostDK[i] *= scale
	}
	if _, err := uploadInto(b, dQ, f32ToBytes(hostDQ)); err != nil {
		return fmt.Errorf("attn recon bwd: scale dQ: %w", err)
	}
	if _, err := uploadInto(b, dK, f32ToBytes(hostDK)); err != nil {
		return fmt.Errorf("attn recon bwd: scale dK: %w", err)
	}
	return nil
}
