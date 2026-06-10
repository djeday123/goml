#!/usr/bin/env python3
"""
QMMA SASS bit-patcher.

Reads a cubin, finds all 16-byte SASS instructions matching the QMMA opcode
(low 11 bits = 0x07a), flips bit 76 to convert
QMMA.16816.F16.E4M3.E4M3  →  QMMA.INVALID2.F16.E4M3.E4M3
and writes a new cubin.

Cubin file format (ELF64): kernel text is in a section starting with
".text.<kernelname>" — its sh_offset points to the raw bytes.

The QMMA opcode signature we look for:
  baseline first byte    = 0x7a    (opcode 0x07a low byte)
  baseline second byte   = 0x?2    (high nibble = control code, low nibble has
                                    opcode high bits)
We refuse to patch unless second byte low nibble is 0x2 (= F16 acc, E4M3×E4M3).

Bit 76 = byte_index 9, bit 4 within that byte.
"""
import sys
import struct
import argparse

QMMA_OP_LO = 0x7a

def find_text_sections(elf_bytes):
    """Find .text.* section ranges in a 64-bit ELF cubin."""
    if elf_bytes[:4] != b"\x7fELF":
        raise SystemExit("not an ELF")
    if elf_bytes[4] != 2:
        raise SystemExit("not ELF64")
    # ELF64 header fields
    e_shoff = struct.unpack_from("<Q", elf_bytes, 0x28)[0]
    e_shentsize = struct.unpack_from("<H", elf_bytes, 0x3a)[0]
    e_shnum = struct.unpack_from("<H", elf_bytes, 0x3c)[0]
    e_shstrndx = struct.unpack_from("<H", elf_bytes, 0x3e)[0]

    def shdr(i):
        off = e_shoff + i * e_shentsize
        sh_name = struct.unpack_from("<I", elf_bytes, off + 0x00)[0]
        sh_type = struct.unpack_from("<I", elf_bytes, off + 0x04)[0]
        sh_flags = struct.unpack_from("<Q", elf_bytes, off + 0x08)[0]
        sh_offset = struct.unpack_from("<Q", elf_bytes, off + 0x18)[0]
        sh_size = struct.unpack_from("<Q", elf_bytes, off + 0x20)[0]
        return sh_name, sh_type, sh_flags, sh_offset, sh_size

    # Section header string table
    _, _, _, str_off, str_size = shdr(e_shstrndx)
    strtab = bytes(elf_bytes[str_off:str_off + str_size])

    def name_of(idx):
        end = strtab.find(b"\x00", idx)
        return strtab[idx:end].decode("ascii", errors="replace")

    sections = []
    for i in range(e_shnum):
        nm_off, sh_type, sh_flags, sh_offset, sh_size = shdr(i)
        nm = name_of(nm_off)
        if nm.startswith(".text"):
            sections.append((nm, sh_offset, sh_size))
    return sections


def patch_qmma(cubin_in, cubin_out, do_patch=True):
    with open(cubin_in, "rb") as f:
        data = bytearray(f.read())

    sections = find_text_sections(data)
    if not sections:
        raise SystemExit("no .text section found")

    total_found = 0
    total_patched = 0
    for nm, off, sz in sections:
        print(f"section {nm}  offset=0x{off:x}  size={sz}")
        # SASS instructions on Blackwell are 16 bytes each.
        for i in range(0, sz, 16):
            ins = data[off + i: off + i + 16]
            if len(ins) < 16:
                break
            # Opcode low byte
            b0 = ins[0]
            if b0 != QMMA_OP_LO:
                continue
            # Refuse anything but F16 acc / E4M3.E4M3 baseline:
            # bits 77,78,79,82-85 all zero  →  byte 9 low nibble == 0x0
            # (bit 75 = byte 9 bit 3; bit 76 = byte 9 bit 4)
            b9 = ins[9]
            # Lower 4 bits of byte9 cover bits 72-75 — control bits we leave alone
            # We need bits 76,77,78,79 = 0 (= 16816, F16, E4M3, E4M3)
            top_nibble = (b9 >> 4) & 0xf
            if top_nibble != 0x0:
                print(f"  @+{i:#06x}  QMMA found but top_nibble=0x{top_nibble:x} (not vanilla 16816.F16.E4M3.E4M3) — skipping")
                continue
            total_found += 1
            if do_patch:
                # Flip bit 76 in this instruction
                data[off + i + 9] = b9 | (1 << 4)
                total_patched += 1
                print(f"  @+{i:#06x}  patched: byte9 0x{b9:02x} → 0x{data[off+i+9]:02x}  (bit 76 set → INVALID2)")
            else:
                print(f"  @+{i:#06x}  match  byte9=0x{b9:02x}  (dry-run)")

    print(f"\nQMMA.16816.F16.E4M3.E4M3 instances: {total_found}")
    if do_patch:
        print(f"Patched: {total_patched}")
        with open(cubin_out, "wb") as f:
            f.write(data)
        print(f"Wrote {cubin_out}")
    return total_found, total_patched


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cubin_in")
    ap.add_argument("cubin_out")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    patch_qmma(args.cubin_in, args.cubin_out, do_patch=not args.dry_run)
