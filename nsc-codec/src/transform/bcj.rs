//! Branch-target transforms for executable code (reversible, in place).
//!
//! A call written as `E8 rel32` encodes the target relative to the next instruction, so calls to
//! the same function from different places look different. Converting the operand to an absolute
//! position (relative to the segment start) makes repeated calls byte-identical, which the context
//! model then predicts well.
//!
//! **x86 (E8 call / E9 jmp):** only operands whose most significant byte is 0x00 or 0xFF are
//! converted (|rel| < 2^24, which covers real branches). The absolute value is computed modulo
//! 2^25 and stored back with an MSB of 0x00/0xFF again. Reversal works because of four facts:
//! * Opcode bytes are never altered.
//! * The scan skips a converted operand, so operand bytes are never read as opcodes.
//! * "MSB is 0x00/0xFF" holds for an operand both before and after conversion.
//! * A declined candidate blocks conversions inside its own operand window.
//! So the decoder makes exactly the encoder's decisions.
//!
//! **ARM64 (BL):** 4-byte words `100101 imm26`; imm26 counts words relative to the instruction and
//! becomes absolute (mod 2^26). Every word is examined, so decisions are trivially shared.

const MASK25: u32 = (1 << 25) - 1;

fn x86(data: &mut [u8], encode: bool) {
    let n = data.len();
    let mut i = 0usize;
    // Positions below `block` may not be converted. A declined candidate at j blocks j+1..=j+4, so
    // the operand byte it was judged on can never be rewritten by a later conversion, and the
    // decoder sees the same byte and declines too.
    let mut block = 0usize;
    while i + 5 <= n {
        let op = data[i];
        let msb = data[i + 4];
        if (op == 0xE8 || op == 0xE9) && !(i >= block && (msb == 0x00 || msb == 0xFF)) {
            block = i + 5;
            i += 1;
            continue;
        }
        if op == 0xE8 || op == 0xE9 {
            let v = u32::from_le_bytes([data[i + 1], data[i + 2], data[i + 3], data[i + 4]]) & MASK25;
            let pos = ((i + 5) as u32) & MASK25;
            let w = if encode { v.wrapping_add(pos) } else { v.wrapping_sub(pos) } & MASK25;
            data[i + 1] = w as u8;
            data[i + 2] = (w >> 8) as u8;
            data[i + 3] = (w >> 16) as u8;
            data[i + 4] = if w & (1 << 24) != 0 { 0xFF } else { 0x00 };
            i += 5;
        } else {
            i += 1;
        }
    }
}

fn arm64(data: &mut [u8], encode: bool) {
    const M26: u32 = (1 << 26) - 1;
    let mut i = 0usize;
    while i + 4 <= data.len() {
        let w = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        if w >> 26 == 0b100101 {
            let pc = ((i / 4) as u32) & M26;
            let imm = w & M26;
            let t = if encode { imm.wrapping_add(pc) } else { imm.wrapping_sub(pc) } & M26;
            data[i..i + 4].copy_from_slice(&((w & !M26) | t).to_le_bytes());
        }
        i += 4;
    }
}

/// x86 by the instruction decoder: the code is read instruction by instruction (as 64-bit code
/// when `wide`), and two kinds of operand that are relative to the end of their instruction
/// become absolute (offsets in the segment, modulo 2^32) — the `rel32` of a call or jump
/// (E8 / E9), and in 64-bit code the `disp32` of `[rip + disp32]`, which is how it reaches
/// every global, string and import.
///
/// Reversal is exact whatever the bytes are (data inside code included): the decoder's reading
/// of an instruction — its length, and where its displacement and immediate are — depends only
/// on prefix, opcode, ModRM and SIB bytes, and only displacement and immediate bytes are ever
/// rewritten. So it reads the transformed code exactly as it read the original.
fn decoded(data: &mut [u8], encode: bool, wide: bool) {
    let mut x = crate::x86::X86::new(wide);
    for i in 0..data.len() {
        let Some(d) = x.update(data[i]) else { continue };
        let end = i + 1;
        let start = end - d.len as usize;
        let at = if d.rip_relative && d.disp_len == 4 {
            start + d.disp_at as usize
        } else if d.call_or_jump && d.imm_len == 4 {
            start + d.imm_at as usize
        } else {
            continue;
        };
        let v = u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]);
        let w = if encode { v.wrapping_add(end as u32) } else { v.wrapping_sub(end as u32) };
        data[at..at + 4].copy_from_slice(&w.to_le_bytes());
    }
}

/// Transform ids (stored in the segment's `xform` byte).
pub const NONE: u8 = 0;
pub const X86: u8 = 1;
pub const ARM64: u8 = 2;
/// Decoder level 2: by the instruction decoder, 64-bit and 32-bit code.
pub const X64: u8 = 3;
pub const X86D: u8 = 4;

/// Which transform suits exe code of `arch` (detector exe params: 1 x86, 2 x86-64, 3 arm64);
/// `level2`: for a decoder of level 2 (x86-64 code then gets its own).
pub fn for_arch(arch: u8, level2: bool) -> u8 {
    // (relocatable code has the flag bit set: its operands are placeholders, left alone)
    match arch { 2 if level2 => X64, 1 if level2 => X86D, 1 | 2 => X86, 3 => ARM64, _ => NONE }
}

pub fn forward(xform: u8, data: &mut [u8]) {
    match xform { X86 => x86(data, true), ARM64 => arm64(data, true), X64 => decoded(data, true, true), X86D => decoded(data, true, false), _ => {} }
}

pub fn inverse(xform: u8, data: &mut [u8]) {
    match xform { X86 => x86(data, false), ARM64 => arm64(data, false), X64 => decoded(data, false, true), X86D => decoded(data, false, false), _ => {} }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(s: &mut u64) -> u64 { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; *s }

    #[test]
    fn inverse_restores_random_and_dense_branch_data() {
        let mut s = 11u64;
        for round in 0..200 {
            let n = (rng(&mut s) % 5000) as usize;
            // bias toward E8/E9 and 00/FF bytes to hit every path, including overlapping candidates
            let data: Vec<u8> = (0..n).map(|_| match rng(&mut s) % 8 { 0 => 0xE8, 1 => 0xE9, 2 => 0x00, 3 => 0xFF, _ => rng(&mut s) as u8 }).collect();
            for x in [X86, ARM64, X64, X86D] {
                let mut t = data.clone();
                forward(x, &mut t);
                inverse(x, &mut t);
                assert!(t == data, "xform {x} round {round} failed to invert");
            }
        }
    }

    #[test]
    fn repeated_calls_become_identical() {
        // two calls to the same target from different places
        let mut code = vec![0x90u8; 64];
        let target = 40i32;
        for &at in &[3usize, 20] {
            let rel = target - (at as i32 + 5);
            code[at] = 0xE8;
            code[at + 1..at + 5].copy_from_slice(&rel.to_le_bytes());
        }
        forward(X86, &mut code);
        assert_eq!(code[4..8], code[21..25]);
    }

    #[test]
    fn real_elf_roundtrip() {
        if let Ok(b) = std::fs::read("/usr/bin/git") {
            for x in [X86, X64, X86D] {
                let mut t = b.clone();
                forward(x, &mut t);
                assert!(t != b);
                inverse(x, &mut t);
                assert!(t == b);
            }
        }
    }

    #[test]
    fn transform_by_architecture() {
        assert_eq!([for_arch(1, false), for_arch(2, false), for_arch(3, false), for_arch(4, false)], [X86, X86, ARM64, NONE]);
        assert_eq!([for_arch(1, true), for_arch(2, true), for_arch(3, true)], [X86D, X64, ARM64]);
        // relocatable code is left alone: its operands are placeholders
        let r = crate::detect::EXE_RELOCATABLE;
        assert_eq!([for_arch(1 | r, true), for_arch(2 | r, true), for_arch(2 | r, false)], [NONE, NONE, NONE]);
    }

    #[test]
    fn rip_relative_references_become_identical() {
        // lea rax, [rip + d] twice, to the same address from two places; a call between them
        let mut code = vec![0x90u8; 64];
        let target = 1000i32;
        for &at in &[2usize, 30] {
            code[at..at + 3].copy_from_slice(&[0x48, 0x8D, 0x05]);
            code[at + 3..at + 7].copy_from_slice(&(target - (at as i32 + 7)).to_le_bytes());
        }
        code[12] = 0xE8;
        code[13..17].copy_from_slice(&(500i32 - 17).to_le_bytes());
        let orig = code.clone();
        forward(X64, &mut code);
        assert_eq!(code[5..9], code[33..37]);
        assert_eq!(code[5..9], 1000u32.to_le_bytes());
        assert_eq!(code[13..17], 500u32.to_le_bytes());
        inverse(X64, &mut code);
        assert!(code == orig);
    }
}
