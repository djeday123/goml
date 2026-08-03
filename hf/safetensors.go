package hf

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
)

// Формат safetensors: 8 байт LE uint64 = длина JSON-заголовка N, затем N байт
// JSON {"tensor.name": {"dtype": "BF16", "shape": [..], "data_offsets": [b,e]},
// ..., "__metadata__": {...}}, затем данные. Оффсеты — относительно начала
// секции данных (8+N). Спецификация: https://github.com/huggingface/safetensors

// TensorInfo — запись одного тензора из заголовка.
type TensorInfo struct {
	Name        string
	Dtype       string
	Shape       []int64
	Begin, End  int64 // относительно начала секции данных
}

// NumElements — произведение размерностей (скаляр = 1).
func (ti *TensorInfo) NumElements() int64 {
	n := int64(1)
	for _, s := range ti.Shape {
		n *= s
	}
	return n
}

// SafeTensors — открытый файл с распарсенным заголовком. Данные читаются
// pread-ом по требованию (файл 2GB+ не грузится в память целиком).
type SafeTensors struct {
	f        *os.File
	dataOff  int64 // 8 + headerLen
	dataLen  int64
	Tensors  map[string]*TensorInfo
	Names    []string // отсортированы по Begin (порядок в файле)
	Metadata map[string]string
}

var dtypeSize = map[string]int64{
	"F64": 8, "F32": 4, "F16": 2, "BF16": 2,
	"I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

// OpenSafeTensors парсит заголовок и валидирует раскладку:
// известные dtype, size(shape)*sizeof(dtype) == end-begin, оффсеты в границах,
// без перекрытий, покрытие секции данных без дыр.
func OpenSafeTensors(path string) (*SafeTensors, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	var hdrLenBuf [8]byte
	if _, err := f.ReadAt(hdrLenBuf[:], 0); err != nil {
		f.Close()
		return nil, fmt.Errorf("чтение длины заголовка: %w", err)
	}
	hdrLen := int64(binary.LittleEndian.Uint64(hdrLenBuf[:]))
	if hdrLen <= 0 || 8+hdrLen > fi.Size() {
		f.Close()
		return nil, fmt.Errorf("длина заголовка %d вне файла %d", hdrLen, fi.Size())
	}
	hdr := make([]byte, hdrLen)
	if _, err := f.ReadAt(hdr, 8); err != nil {
		f.Close()
		return nil, fmt.Errorf("чтение заголовка: %w", err)
	}
	var rawEntries map[string]json.RawMessage
	if err := json.Unmarshal(hdr, &rawEntries); err != nil {
		f.Close()
		return nil, fmt.Errorf("JSON заголовка: %w", err)
	}
	st := &SafeTensors{
		f:       f,
		dataOff: 8 + hdrLen,
		dataLen: fi.Size() - 8 - hdrLen,
		Tensors: make(map[string]*TensorInfo),
	}
	for name, raw := range rawEntries {
		if name == "__metadata__" {
			if err := json.Unmarshal(raw, &st.Metadata); err != nil {
				f.Close()
				return nil, fmt.Errorf("__metadata__: %w", err)
			}
			continue
		}
		var e struct {
			Dtype       string  `json:"dtype"`
			Shape       []int64 `json:"shape"`
			DataOffsets [2]int64 `json:"data_offsets"`
		}
		if err := json.Unmarshal(raw, &e); err != nil {
			f.Close()
			return nil, fmt.Errorf("тензор %q: %w", name, err)
		}
		ti := &TensorInfo{Name: name, Dtype: e.Dtype, Shape: e.Shape,
			Begin: e.DataOffsets[0], End: e.DataOffsets[1]}
		sz, ok := dtypeSize[ti.Dtype]
		if !ok {
			f.Close()
			return nil, fmt.Errorf("тензор %q: неизвестный dtype %q", name, ti.Dtype)
		}
		if ti.Begin < 0 || ti.End < ti.Begin || ti.End > st.dataLen {
			f.Close()
			return nil, fmt.Errorf("тензор %q: оффсеты [%d,%d) вне секции данных %d", name, ti.Begin, ti.End, st.dataLen)
		}
		if want := ti.NumElements() * sz; want != ti.End-ti.Begin {
			f.Close()
			return nil, fmt.Errorf("тензор %q: shape %v * %s = %d байт, а оффсеты дают %d", name, ti.Shape, ti.Dtype, want, ti.End-ti.Begin)
		}
		st.Tensors[name] = ti
		st.Names = append(st.Names, name)
	}
	sort.Slice(st.Names, func(i, j int) bool {
		return st.Tensors[st.Names[i]].Begin < st.Tensors[st.Names[j]].Begin
	})
	// Непрерывность: тензоры покрывают [0, dataLen) без дыр и перекрытий.
	var cursor int64
	for _, n := range st.Names {
		ti := st.Tensors[n]
		if ti.Begin != cursor {
			f.Close()
			return nil, fmt.Errorf("тензор %q: begin %d != ожидаемому %d (дыра/перекрытие)", n, ti.Begin, cursor)
		}
		cursor = ti.End
	}
	if cursor != st.dataLen {
		f.Close()
		return nil, fmt.Errorf("хвост секции данных: покрыто %d из %d", cursor, st.dataLen)
	}
	return st, nil
}

// Close закрывает файл.
func (st *SafeTensors) Close() error { return st.f.Close() }

// ReadRaw читает сырые байты тензора pread-ом.
func (st *SafeTensors) ReadRaw(name string) ([]byte, error) {
	ti, ok := st.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("тензор %q отсутствует", name)
	}
	buf := make([]byte, ti.End-ti.Begin)
	if _, err := st.f.ReadAt(buf, st.dataOff+ti.Begin); err != nil {
		return nil, fmt.Errorf("тензор %q: %w", name, err)
	}
	return buf, nil
}

// ReadF32 читает тензор и конвертирует в []float32.
// BF16->F32 и F16->F32 точны (расширение мантиссы без потерь).
func (st *SafeTensors) ReadF32(name string) ([]float32, error) {
	ti, ok := st.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("тензор %q отсутствует", name)
	}
	raw, err := st.ReadRaw(name)
	if err != nil {
		return nil, err
	}
	n := int(ti.NumElements())
	out := make([]float32, n)
	switch ti.Dtype {
	case "F32":
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "BF16":
		for i := 0; i < n; i++ {
			out[i] = BF16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "F16":
		for i := 0; i < n; i++ {
			out[i] = F16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	default:
		return nil, fmt.Errorf("тензор %q: ReadF32 не поддерживает dtype %s", name, ti.Dtype)
	}
	return out, nil
}

// ReadF64 — как ReadF32, но в []float64 (F32->F64 точен).
func (st *SafeTensors) ReadF64(name string) ([]float64, error) {
	f32, err := st.ReadF32(name)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(f32))
	for i, v := range f32 {
		out[i] = float64(v)
	}
	return out, nil
}

// BF16ToF32 — точная конверсия: bf16 = верхние 16 бит f32.
func BF16ToF32(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

// F16ToF32 — точная конверсия IEEE 754 half -> single (с денормалами, Inf, NaN).
func F16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1F
	man := uint32(h) & 0x3FF
	switch {
	case exp == 0x1F: // Inf / NaN
		return math.Float32frombits(sign | 0xFF<<23 | man<<13)
	case exp != 0: // нормальное
		return math.Float32frombits(sign | (exp-15+127)<<23 | man<<13)
	case man == 0: // ±0
		return math.Float32frombits(sign)
	default: // денормал: value = man * 2^-24; нормализуем под f32
		shift := uint32(0)
		for man&0x400 == 0 {
			man <<= 1
			shift++
		}
		man &= 0x3FF
		return math.Float32frombits(sign | (113-shift)<<23 | man<<13)
	}
}
