package abjexam

// A-LLM-6 П.4а: BattleASnapScratch — per-layer снапшоты боевого пути в
// ОТДЕЛЬНОМ scratch (НЕ в BattleAScratch — [[feedback-fa-fwd-scratch-alloc-
// instability]]: аллокации в общем scratch триггерили FA-нестабильность).
// Зеркало снапшот-дисциплины bwdBattleAF32 (caveat-1 класс закрыт).

import (
	"time"

	"github.com/djeday123/goml/backend"
)

type BattleASnapScratch struct {
	XPreAttn []backend.Storage // F32 [M, D] x L — X до RMSNorm1
	XPreFFN  []backend.Storage // F32 [M, D] x L — X до RMSNorm2
	QPerm    []backend.Storage // F32 [BH, S, HD] x L — post-RoPE
	KPerm    []backend.Storage // F32 [BH, S, HD] x L — post-RoPE
	VPerm    []backend.Storage // F32 [BH, S, HD] x L
	OAttn    []backend.Storage // F32 [BH, S, HD] x L — O_real (после cast+scaleV)
	XPreTop  backend.Storage   // F32 [M, D]
	// Пятая карта блоков (П.4г): staging-цена FA-блока ОТДЕЛЬНОЙ строкой.
	TFABlockKernels time.Duration // квант+fa-fwd+D+merged+dk+dq (GPU-час блока)
	TFABlockStaging time.Duration // host-staging D2H/H2D блока (cert-плата)
}

func NewBattleASnapScratch(cfg BattleACfg, b backend.Backend) (*BattleASnapScratch, error) {
	M := cfg.B * cfg.S
	BH := cfg.B * cfg.H
	sn := &BattleASnapScratch{}
	al := func(bytes int) (backend.Storage, error) { return b.Alloc(bytes) }
	mk := func(n int, bytes int) ([]backend.Storage, error) {
		out := make([]backend.Storage, n)
		for i := 0; i < n; i++ {
			s, err := al(bytes)
			if err != nil {
				return nil, err
			}
			out[i] = s
		}
		return out, nil
	}
	var err error
	if sn.XPreAttn, err = mk(cfg.L, M*cfg.D*4); err != nil {
		return nil, err
	}
	if sn.XPreFFN, err = mk(cfg.L, M*cfg.D*4); err != nil {
		return nil, err
	}
	if sn.QPerm, err = mk(cfg.L, BH*cfg.S*cfg.HD*4); err != nil {
		return nil, err
	}
	if sn.KPerm, err = mk(cfg.L, BH*cfg.S*cfg.HD*4); err != nil {
		return nil, err
	}
	if sn.VPerm, err = mk(cfg.L, BH*cfg.S*cfg.HD*4); err != nil {
		return nil, err
	}
	if sn.OAttn, err = mk(cfg.L, BH*cfg.S*cfg.HD*4); err != nil {
		return nil, err
	}
	if sn.XPreTop, err = al(M * cfg.D * 4); err != nil {
		return nil, err
	}
	return sn, nil
}

func (sn *BattleASnapScratch) FreeAll(b backend.Backend) {
	if sn == nil {
		return
	}
	for _, arr := range [][]backend.Storage{sn.XPreAttn, sn.XPreFFN, sn.QPerm, sn.KPerm, sn.VPerm, sn.OAttn} {
		for _, s := range arr {
			if s != nil {
				b.Free(s)
			}
		}
	}
	if sn.XPreTop != nil {
		b.Free(sn.XPreTop)
	}
}

func timeNow() time.Time              { return time.Now() }
func timeSince(t time.Time) time.Duration { return time.Since(t) }
