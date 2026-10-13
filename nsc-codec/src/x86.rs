//! An x86 / x86-64 instruction decoder, one byte at a time: just enough to know what each byte
//! of machine code *is* — prefix, opcode, ModRM, SIB, displacement or immediate — and which
//! instruction it belongs to.
//!
//! The executable model conditions its predictions on that ([`X86::context`]), and the branch
//! transform uses it to find the operands worth rewriting. Nothing is executed or validated: on
//! data inside code (jump tables, padding, literals) the decoder reads nonsense and falls back
//! in step within a few instructions, as x86 code does. A wrong guess costs compression, never
//! correctness: both sides run the same decoder over the same bytes.

/// What the next byte is.
pub mod role {
    /// A prefix, or the (first) opcode byte.
    pub const START: u8 = 0;
    /// The second opcode byte (after 0F); the third (after 0F 38, 0F 3A, or a VEX prefix).
    pub const OP2: u8 = 1;
    pub const OP3: u8 = 2;
    /// A VEX / EVEX prefix's payload.
    pub const VEX: u8 = 3;
    pub const MODRM: u8 = 4;
    pub const SIB: u8 = 5;
    pub const DISP: u8 = 6;
    pub const IMM: u8 = 7;
}

// Opcode properties: a ModRM byte follows; the kind of immediate; the operation is chosen by
// ModRM's middle field; the immediate is a branch distance.
const M: u8 = 1;
const I8: u8 = 1 << 1;
const IZ: u8 = 2 << 1;
const I16: u8 = 3 << 1;
const IV: u8 = 4 << 1;
const ENTER: u8 = 5 << 1;
const FAR: u8 = 6 << 1;
const MOFFS: u8 = 7 << 1;
const IMM: u8 = 7 << 1;
const GRP: u8 = 1 << 4;
const REL: u8 = 1 << 5;
const PREFIX: u8 = 1 << 6;

const fn range(mut t: [u8; 256], from: usize, to: usize, v: u8) -> [u8; 256] {
    let mut i = from;
    while i <= to {
        t[i] = v;
        i += 1;
    }
    t
}

/// The one-byte opcodes.
const fn one_byte() -> [u8; 256] {
    let mut t = [0u8; 256];
    // the eight arithmetic rows: r/m forms, AL imm8, eAX imm
    let mut row = 0;
    while row < 0x40 {
        t = range(t, row, row + 3, M);
        t[row + 4] = I8;
        t[row + 5] = IZ;
        row += 8;
    }
    let prefixes = [0x26, 0x2E, 0x36, 0x3E, 0x64, 0x65, 0x66, 0x67, 0xF0, 0xF2, 0xF3];
    let mut i = 0;
    while i < prefixes.len() {
        t[prefixes[i]] = PREFIX;
        i += 1;
    }
    t[0x62] = M;
    t[0x63] = M;
    t[0x68] = IZ;
    t[0x69] = M | IZ;
    t[0x6A] = I8;
    t[0x6B] = M | I8;
    t = range(t, 0x70, 0x7F, I8 | REL);
    t[0x80] = M | I8 | GRP;
    t[0x81] = M | IZ | GRP;
    t[0x82] = M | I8 | GRP;
    t[0x83] = M | I8 | GRP;
    t = range(t, 0x84, 0x8E, M);
    t[0x8F] = M | GRP;
    t[0x9A] = FAR;
    t = range(t, 0xA0, 0xA3, MOFFS);
    t[0xA8] = I8;
    t[0xA9] = IZ;
    t = range(t, 0xB0, 0xB7, I8);
    t = range(t, 0xB8, 0xBF, IV);
    t[0xC0] = M | I8 | GRP;
    t[0xC1] = M | I8 | GRP;
    t[0xC2] = I16;
    t[0xC4] = M;
    t[0xC5] = M;
    t[0xC6] = M | I8 | GRP;
    t[0xC7] = M | IZ | GRP;
    t[0xC8] = ENTER;
    t[0xCA] = I16;
    t[0xCD] = I8;
    t = range(t, 0xD0, 0xD3, M | GRP);
    t[0xD4] = I8;
    t[0xD5] = I8;
    t = range(t, 0xD8, 0xDF, M | GRP);
    t = range(t, 0xE0, 0xE3, I8 | REL);
    t = range(t, 0xE4, 0xE7, I8);
    t[0xE8] = IZ | REL;
    t[0xE9] = IZ | REL;
    t[0xEA] = FAR;
    t[0xEB] = I8 | REL;
    // F6 / F7: an immediate only for TEST (decided when ModRM is read)
    t[0xF6] = M | GRP;
    t[0xF7] = M | GRP;
    t[0xFE] = M | GRP;
    t[0xFF] = M | GRP;
    t
}

/// The opcodes after 0F.
const fn two_byte() -> [u8; 256] {
    let mut t = [0u8; 256];
    t[0x00] = M | GRP;
    t[0x01] = M | GRP;
    t[0x02] = M;
    t[0x03] = M;
    t[0x0D] = M;
    t[0x0F] = M | I8;
    t = range(t, 0x10, 0x17, M);
    t[0x18] = M | GRP;
    t = range(t, 0x19, 0x1F, M);
    t = range(t, 0x20, 0x23, M);
    t = range(t, 0x28, 0x2F, M);
    t = range(t, 0x40, 0x6F, M);
    t[0x70] = M | I8;
    t = range(t, 0x71, 0x73, M | I8 | GRP);
    t = range(t, 0x74, 0x76, M);
    t = range(t, 0x78, 0x7F, M);
    t = range(t, 0x80, 0x8F, IZ | REL);
    t = range(t, 0x90, 0x9F, M);
    t[0xA3] = M;
    t[0xA4] = M | I8;
    t[0xA5] = M;
    t[0xAB] = M;
    t[0xAC] = M | I8;
    t[0xAD] = M;
    t[0xAE] = M | GRP;
    t[0xAF] = M;
    t = range(t, 0xB0, 0xB9, M);
    t[0xBA] = M | I8 | GRP;
    t = range(t, 0xBB, 0xC1, M);
    t[0xC2] = M | I8;
    t[0xC3] = M;
    t = range(t, 0xC4, 0xC6, M | I8);
    t[0xC7] = M | GRP;
    t = range(t, 0xD0, 0xFF, M);
    t
}

static ONE: [u8; 256] = one_byte();
static TWO: [u8; 256] = two_byte();

/// What kind of thing an instruction does (16 classes): a coarser identity, so longer runs of
/// instructions can be told apart.
mod cat {
    pub const OTHER: u8 = 0;
    pub const MOV: u8 = 1;
    pub const LEA: u8 = 2;
    pub const ARITH: u8 = 3;
    pub const LOGIC: u8 = 4;
    pub const CMP: u8 = 5;
    pub const SHIFT: u8 = 6;
    pub const PUSH: u8 = 7;
    pub const POP: u8 = 8;
    pub const JCC: u8 = 9;
    pub const JMP: u8 = 10;
    pub const CALL: u8 = 11;
    pub const RET: u8 = 12;
    pub const PAD: u8 = 13;
    pub const COND: u8 = 14;
    pub const VECTOR: u8 = 15;
}

/// The class of the instruction with opcode `op` in `map`, `reg` its ModRM middle field.
fn category(map: u8, op: u8, reg: u8, vex: bool) -> u8 {
    use cat::*;
    if vex || map >= 2 { return VECTOR; }
    if map == 1 {
        return match op {
            0x80..=0x8F => JCC,
            0x40..=0x4F | 0x90..=0x9F => COND,
            0xB6 | 0xB7 | 0xBE | 0xBF => MOV,
            0xAF => ARITH,
            0xA4 | 0xA5 | 0xAC | 0xAD => SHIFT,
            0x1F | 0x0B => PAD,
            0xA3 | 0xAB | 0xB3 | 0xBB | 0xBA | 0xBC | 0xBD => LOGIC,
            0x05 => CALL,
            _ => VECTOR,
        };
    }
    match op {
        0x00..=0x3F if op & 7 < 6 => match op >> 3 { 0 | 2 | 3 | 5 => ARITH, 7 => CMP, _ => LOGIC },
        0x06 | 0x0E | 0x16 | 0x1E | 0x50..=0x57 | 0x68 | 0x6A | 0x60 | 0x9C => PUSH,
        0x07 | 0x17 | 0x1F | 0x58..=0x5F | 0x8F | 0x61 | 0x9D => POP,
        0x40..=0x4F => ARITH,
        0x63 | 0x88..=0x8C | 0x8E | 0xA0..=0xA3 | 0xB0..=0xBF | 0xC6 | 0xC7 => MOV,
        0x69 | 0x6B => ARITH,
        0x70..=0x7F | 0xE0..=0xE3 => JCC,
        0x80..=0x83 => match reg { 7 => CMP, 1 | 4 | 6 => LOGIC, _ => ARITH },
        0x84 | 0x85 | 0xA8 | 0xA9 => CMP,
        0x86 | 0x87 | 0x91..=0x97 | 0x98 | 0x99 => OTHER,
        0x8D => LEA,
        0x90 | 0xCC | 0xF4 => PAD,
        0xC0 | 0xC1 | 0xD0..=0xD3 => SHIFT,
        0xC2 | 0xC3 | 0xCA | 0xCB | 0xCF | 0xC9 => RET,
        0xD8..=0xDF => VECTOR,
        0xE8 | 0x9A => CALL,
        0xE9 | 0xEA | 0xEB => JMP,
        0xF6 | 0xF7 => match reg { 0 | 1 => CMP, 2 => LOGIC, _ => ARITH },
        0xFE => ARITH,
        0xFF => match reg { 0 | 1 => ARITH, 2 | 3 => CALL, 4 | 5 => JMP, _ => PUSH },
        _ => OTHER,
    }
}

/// Developer ablation: the contexts in use (bit per context; all by default).
pub static KEEP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// Is context `n` in use?
pub fn kept(n: u32) -> bool { KEEP.load(std::sync::atomic::Ordering::Relaxed) >> n & 1 == 1 }

/// Prefix flags of the instruction being read.
pub mod pfx {
    pub const OPSIZE: u8 = 1;
    pub const ADDRSIZE: u8 = 2;
    pub const REP: u8 = 4;
    pub const REPNE: u8 = 8;
    pub const SEGMENT: u8 = 16;
    pub const LOCK: u8 = 32;
    pub const VEX: u8 = 64;
}

/// An instruction just completed (for [`X86::update`]'s caller).
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Done {
    /// Its length, and where in it the displacement and the immediate start (0: none).
    pub len: u8,
    pub disp_at: u8,
    pub disp_len: u8,
    pub imm_at: u8,
    pub imm_len: u8,
    /// The displacement is relative to the end of the instruction (x86-64 `[rip + disp32]`).
    pub rip_relative: bool,
    /// The immediate is a branch distance; a call or an unconditional jump (E8 / E9).
    pub relative: bool,
    pub call_or_jump: bool,
}

#[derive(Clone)]
pub struct X86 {
    wide: bool,
    // the instruction being read: the role of the next byte, the bytes of its field read so
    // far, and how many are still to come
    role: u8,
    k: u8,
    left: u8,
    len: u8,
    prefixes: u8,
    rex: u8,
    /// Opcode map (0 one byte, 1 after 0F, 2 after 0F 38, 3 after 0F 3A) and byte, once read.
    map: u8,
    op: u8,
    have_op: bool,
    flags: u8,
    modrm: u8,
    have_modrm: bool,
    sib: u8,
    disp: u8,
    imm: u8,
    /// The bytes of the field being read, the first one lowest.
    field: u64,
    vex: u8,
    cur: Done,
    // what came before: the last instructions' identities, the last ModRM and REX bytes; the
    // last displacement by base register
    prev: [u32; 4],
    prev_modrm: u8,
    prev_rex: u8,
    last_base: [u32; 16],
    /// The classes of the last instructions (4 bits each, the latest lowest).
    cats: u32,
    /// By instruction: the last value of its field.
    by_op: Vec<u32>,
}

const FIELDS: usize = 1 << 12;

impl X86 {
    /// A decoder for 32-bit code, or (`wide`) for x86-64.
    pub fn new(wide: bool) -> Self {
        X86 {
            wide, role: role::START, k: 0, left: 0, len: 0, prefixes: 0, rex: 0, map: 0, op: 0, have_op: false, flags: 0, modrm: 0, have_modrm: false,
            sib: 0, disp: 0, imm: 0, field: 0, vex: 0, cur: Done::default(), prev: [0; 4], prev_modrm: 0, prev_rex: 0, last_base: [0; 16],
            cats: 0, by_op: vec![0; FIELDS],
        }
    }

    /// The role of the next byte, and how many bytes of its field came before it.
    #[inline]
    pub fn role(&self) -> (u8, u8) { (self.role, self.k) }

    /// The instruction's identity so far: opcode map and byte, the mandatory-prefix class, REX.W;
    /// and for the opcodes whose operation is in ModRM, that field (once read).
    fn ident(&self) -> u32 {
        if !self.have_op { return 0; }
        let group = if self.flags & GRP != 0 && self.have_modrm { 8 | (self.modrm >> 3 & 7) as u32 } else { 0 };
        let p = self.prefixes;
        let class = if p & pfx::OPSIZE != 0 { 1 } else if p & pfx::REP != 0 { 2 } else if p & pfx::REPNE != 0 { 3 } else { 0 };
        1 + ((self.map as u32) << 8 | self.op as u32) | group << 11 | class << 15 | ((self.rex >> 3 & 1) as u32) << 17 | ((p & pfx::VEX != 0) as u32) << 18
    }

    /// The base register of the addressing form (with REX.B), 16 for none.
    fn base(&self) -> usize {
        if !self.have_modrm || self.modrm >> 6 == 3 { return 16; }
        let rm = if self.modrm & 7 == 4 && self.role != role::SIB { self.sib & 7 } else { self.modrm & 7 };
        (rm | (self.rex & 1) << 3) as usize
    }

    /// Where the last value of this instruction's field (displacement or immediate) is remembered.
    fn op_slot(&self) -> usize { ((self.ident() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - 12)) as usize }

    /// Context `n` (0..[`CONTEXTS`]) for the next byte; `c4` the last four bytes, `pos` the offset in
    /// the code.
    pub fn context(&self, n: u32, c4: u32, pos: usize) -> u64 {
        let rk = (self.role as u64) << 3 | self.k.min(7) as u64;
        let id = self.ident() as u64;
        let m = if self.have_modrm { 0x100 | self.modrm as u64 } else { 0 };
        let p = [self.prev[0] as u64, self.prev[1] as u64, self.prev[2] as u64];
        let field = self.field & 0xFF_FFFF;
        let cats = self.cats as u64;
        // (chosen by greedy elimination from twice as many: `bench/exe_ablate.py`)
        match n {
            // the instruction so far; with the one before
            0 => rk | id << 7 | m << 27 | (self.prefixes as u64) << 36,
            1 => rk | id << 7 | m << 27 | p[0] << 36,
            // the field being read: its bytes so far
            2 => rk | id << 7 | (m & 0x1C7) << 27 | field << 36,
            // a displacement: the last one off the same base register
            3 => {
                let b = self.base();
                rk | (b as u64) << 7 | (self.last_base[b & 15] as u64 & 0xFFFF) << 12 | (field & 0xFFFF) << 28 | ((self.modrm >> 6) as u64) << 44
            }
            // the registers in play: REX of this instruction and the last, the byte before
            4 => rk | id << 7 | (self.rex as u64) << 27 | (self.prev_rex as u64) << 35 | (c4 as u64 & 0xFF) << 43,
            // the three instructions before; the classes of the last six; a hash of the last four
            5 => rk | p[0] << 7 | p[1] << 27 | (p[2] & 0xFFFF) << 47,
            6 => rk | id << 7 | (cats & 0xFF_FFFF) << 27,
            7 => rk | (p[0] ^ p[1].wrapping_mul(0x9E37_79B1) ^ p[2].wrapping_mul(0x85EB_CA6B) ^ (self.prev[3] as u64).wrapping_mul(0xC2B2_AE35)) << 7,
            // where in a 16-byte row (functions start on them, padding ends on them)
            8 => rk | (pos as u64 & 15) << 7 | p[0] << 11 | id << 31,
            9 => rk | id << 7 | m << 27 | (pos as u64 & 15) << 36,
            // the byte before
            10 => rk | id << 7 | (c4 as u64 & 0xFF) << 27,
            // the instruction before with its operands
            11 => rk | p[0] << 7 | (self.prev_modrm as u64) << 27 | (p[1] & 0xFFFF) << 35 | (id & 0xFFF) << 51,
            // the field's last value for this instruction; its bytes so far after these classes
            12 => rk | id << 7 | (self.by_op[self.op_slot()] as u64 & 0xFFFF) << 27 | (field & 0xFF) << 43,
            _ => rk | id << 7 | (cats & 0xFF) << 27 | field << 35,
        }
    }

    /// Another: the opcode byte, once read (255: not yet).
    pub fn opcode_class(&self) -> usize { if self.have_op { self.op as usize } else { 255 } }

    /// A small number for mixer weight sets and the like: the role, the place in the field, and
    /// whether the instruction addresses memory.
    pub fn state(&self) -> usize {
        let mem = (self.have_modrm && self.modrm >> 6 != 3) as usize;
        ((self.role as usize) << 3 | self.k.min(7) as usize) << 1 | mem
    }

    fn finish(&mut self) -> Done {
        let done = Done { len: self.len, ..self.cur };
        let c = category(self.map, self.op, self.modrm >> 3 & 7, self.prefixes & pfx::VEX != 0);
        self.cats = self.cats << 4 | c as u32;
        self.prev = [self.ident(), self.prev[0], self.prev[1], self.prev[2]];
        self.prev_modrm = if self.have_modrm { self.modrm } else { 0 };
        self.prev_rex = self.rex;
        (self.role, self.k, self.left, self.len, self.prefixes, self.rex, self.map, self.have_op, self.have_modrm) = (role::START, 0, 0, 0, 0, 0, 0, false, false);
        (self.disp, self.imm, self.field, self.vex, self.sib, self.cur) = (0, 0, 0, 0, 0, Done::default());
        done
    }

    /// After the opcode and addressing bytes: on to the displacement, the immediate, or the end.
    fn next_field(&mut self) -> Option<Done> {
        self.k = 0;
        self.field = 0;
        if self.role < role::DISP && self.disp > 0 {
            (self.role, self.left) = (role::DISP, self.disp);
            (self.cur.disp_at, self.cur.disp_len) = (self.len, self.disp);
            return None;
        }
        if self.role < role::IMM && self.imm > 0 {
            (self.role, self.left) = (role::IMM, self.imm);
            (self.cur.imm_at, self.cur.imm_len) = (self.len, self.imm);
            return None;
        }
        Some(self.finish())
    }

    fn opcode(&mut self, map: u8, b: u8) -> Option<Done> {
        (self.map, self.op, self.have_op) = (map, b, true);
        let f = match map { 0 => ONE[b as usize], 1 => TWO[b as usize], 2 => M, _ => M | I8 };
        self.flags = f;
        let narrow = self.prefixes & pfx::OPSIZE != 0 && self.rex & 8 == 0;
        let short = self.prefixes & pfx::ADDRSIZE != 0;
        self.imm = match f & IMM {
            I8 => 1,
            // a near branch in 64-bit code is always 32 bits
            IZ => if narrow && !(self.wide && f & REL != 0) { 2 } else { 4 },
            I16 => 2,
            IV => if self.rex & 8 != 0 { 8 } else if narrow { 2 } else { 4 },
            ENTER => 3,
            FAR => if narrow { 4 } else { 6 },
            MOFFS => match (self.wide, short) { (true, false) => 8, (true, true) | (false, false) => 4, (false, true) => 2 },
            _ => 0,
        };
        self.cur.relative = f & REL != 0;
        self.cur.call_or_jump = map == 0 && (b == 0xE8 || b == 0xE9);
        if f & M != 0 {
            (self.role, self.k) = (role::MODRM, 0);
            return None;
        }
        self.role = role::SIB; // nothing to address: straight to the immediate
        self.next_field()
    }

    /// Take in the next byte of code. Returns the instruction it completes, if it does.
    pub fn update(&mut self, b: u8) -> Option<Done> {
        self.len = self.len.saturating_add(1);
        match self.role {
            role::START => {
                if ONE[b as usize] & PREFIX != 0 && self.k < 14 {
                    self.prefixes |= match b { 0x66 => pfx::OPSIZE, 0x67 => pfx::ADDRSIZE, 0xF3 => pfx::REP, 0xF2 => pfx::REPNE, 0xF0 => pfx::LOCK, _ => pfx::SEGMENT };
                    self.k += 1;
                    return None;
                }
                if self.wide && b & 0xF0 == 0x40 && self.k < 14 {
                    self.rex = b;
                    self.k += 1;
                    return None;
                }
                match b {
                    0x0F => (self.role, self.k) = (role::OP2, 0),
                    // VEX and EVEX prefixes (in 32-bit code these bytes are also LES, LDS and
                    // BOUND, told apart by the byte after: taken as those)
                    0xC5 if self.wide => (self.role, self.k, self.left, self.vex, self.map) = (role::VEX, 0, 1, b, 1),
                    0xC4 | 0x62 if self.wide => (self.role, self.k, self.left, self.vex) = (role::VEX, 0, if b == 0xC4 { 2 } else { 3 }, b),
                    _ => return self.opcode(0, b),
                }
                None
            }
            role::OP2 => match b {
                0x38 | 0x3A => {
                    (self.role, self.map) = (role::OP3, if b == 0x38 { 2 } else { 3 });
                    None
                }
                _ => self.opcode(1, b),
            },
            role::OP3 => self.opcode(self.map.clamp(1, 3), b),
            role::VEX => {
                self.prefixes |= pfx::VEX;
                if self.k == 0 && self.vex != 0xC5 { self.map = b & 3; }
                // the last payload byte holds the implied prefix; the one with W differs
                if self.left == 1 || (self.vex == 0x62 && self.k == 1) {
                    self.prefixes |= match b & 3 { 1 => pfx::OPSIZE, 2 => pfx::REP, 3 => pfx::REPNE, _ => 0 };
                }
                if self.vex != 0xC5 && self.k == 1 && b & 0x80 != 0 { self.rex |= 8; }
                self.k += 1;
                self.left -= 1;
                if self.left == 0 { (self.role, self.k) = (role::OP3, 0); }
                None
            }
            role::MODRM => {
                (self.modrm, self.have_modrm) = (b, true);
                let (md, reg, rm) = (b >> 6, b >> 3 & 7, b & 7);
                let mut sib = false;
                if !self.wide && self.prefixes & pfx::ADDRSIZE != 0 {
                    self.disp = match md { 0 if rm == 6 => 2, 1 => 1, 2 => 2, _ => 0 };
                } else {
                    sib = md != 3 && rm == 4;
                    self.disp = match md { 0 if rm == 5 => 4, 1 => 1, 2 => 4, _ => 0 };
                    self.cur.rip_relative = self.wide && md == 0 && rm == 5;
                }
                // TEST r/m, imm among the unary group
                if self.map == 0 && self.op & 0xFE == 0xF6 && reg < 2 {
                    self.imm = if self.op == 0xF6 { 1 } else if self.prefixes & pfx::OPSIZE != 0 && self.rex & 8 == 0 { 2 } else { 4 };
                }
                if sib {
                    (self.role, self.k) = (role::SIB, 0);
                    return None;
                }
                self.role = role::SIB;
                self.next_field()
            }
            role::SIB => {
                self.sib = b;
                if self.modrm >> 6 == 0 && b & 7 == 5 { self.disp = 4; }
                self.next_field()
            }
            _ => {
                self.field |= (b as u64) << (8 * self.k.min(7));
                self.k += 1;
                self.left -= 1;
                if self.left > 0 { return None; }
                // remember the value: by instruction, and a displacement by its base register
                let op = self.op_slot();
                self.by_op[op] = self.field as u32;
                if self.role == role::DISP {
                    let b = self.base();
                    if b < 16 { self.last_base[b] = self.field as u32; }
                }
                self.next_field()
            }
        }
    }
}

/// Contexts [`X86::context`] offers.
pub const CONTEXTS: u32 = 14;

/// Test material: x86-64 machine code as a compiler might write it — `funcs` functions with
/// frames, stack slots, references to a few globals through `[rip + disp32]`, calls to earlier
/// functions, compare-and-branch — padded to 16-byte rows.
#[cfg(test)]
pub fn sample_code(funcs: usize) -> Vec<u8> {
    let mut s = 0x1234_5678_9ABC_DEFu64;
    let mut rng = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let mut c: Vec<u8> = vec![];
    let mut starts = vec![0usize];
    let rel = |c: &mut Vec<u8>, target: usize| { let d = target as i64 - (c.len() as i64 + 4); c.extend_from_slice(&(d as i32).to_le_bytes()); };
    for _ in 0..funcs {
        while c.len() % 16 != 0 { c.push(0xCC); }
        starts.push(c.len());
        c.extend_from_slice(&[0x55, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x10 + 16 * (rng() % 4) as u8]);
        for _ in 0..4 + rng() % 12 {
            match rng() % 6 {
                0 => c.extend_from_slice(&[0x48, 0x89, 0x7D, (-8 * (1 + rng() as i64 % 6)) as u8]),
                1 => {
                    c.extend_from_slice(&[0x48, 0x8D, 0x05]);
                    rel(&mut c, 200_000 + 64 * (rng() % 24) as usize);
                }
                2 => {
                    c.push(0xE8);
                    let t = starts[(rng() % starts.len() as u64) as usize];
                    rel(&mut c, t);
                }
                3 => c.extend_from_slice(&[0x8B, 0x44, 0x88, 0x10]),
                4 => c.extend_from_slice(&[0x83, 0xF8, (rng() % 9) as u8, 0x75, 4 + (rng() % 5) as u8]),
                _ => {
                    c.push(0xBF);
                    c.extend_from_slice(&((rng() % 40) as u32).to_le_bytes());
                }
            }
        }
        c.extend_from_slice(&[0xC9, 0xC3]);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lengths of the instructions in `code`.
    fn lengths(wide: bool, code: &[u8]) -> Vec<u8> {
        let mut x = X86::new(wide);
        code.iter().filter_map(|&b| x.update(b)).map(|d| d.len).collect()
    }

    #[test]
    fn decodes_common_instructions() {
        // 64-bit: push rbp; mov rbp, rsp; sub rsp, 0x20; mov [rbp-8], rdi; lea rax, [rip+0x1234];
        // call rel32; mov eax, [rax+rcx*4+0x10]; movabs rax, imm64; jne rel8; ret; nopw 0(%rax,%rax,1)
        let code: &[u8] = &[
            0x55, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x20, 0x48, 0x89, 0x7D, 0xF8, 0x48, 0x8D, 0x05, 0x34, 0x12, 0x00, 0x00, 0xE8, 1, 2, 3, 4, 0x8B, 0x44, 0x88, 0x10, 0x48,
            0xB8, 1, 2, 3, 4, 5, 6, 7, 8, 0x75, 0x05, 0xC3, 0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00,
        ];
        assert_eq!(lengths(true, code), [1, 3, 4, 4, 7, 5, 4, 10, 2, 1, 6]);
        // what the fields are
        let mut x = X86::new(true);
        let done: Vec<Done> = code.iter().filter_map(|&b| x.update(b)).collect();
        assert_eq!(done[4], Done { len: 7, disp_at: 3, disp_len: 4, rip_relative: true, ..Default::default() });
        assert_eq!(done[5], Done { len: 5, imm_at: 1, imm_len: 4, relative: true, call_or_jump: true, ..Default::default() });
        assert_eq!(done[7], Done { len: 10, imm_at: 2, imm_len: 8, ..Default::default() });
        assert_eq!(done[2], Done { len: 4, imm_at: 3, imm_len: 1, ..Default::default() });
        // 32-bit: mov eax, [0x12345678]; test byte [ebx], 1; inc eax; imul eax, [esi+4], 100000;
        // jz rel32; push imm32; movzx eax, byte [ecx]; enter 16, 0
        let code: &[u8] = &[
            0xA1, 0x78, 0x56, 0x34, 0x12, 0xF6, 0x03, 0x01, 0x40, 0x69, 0x46, 0x04, 0xA0, 0x86, 0x01, 0x00, 0x0F, 0x84, 1, 2, 3, 4, 0x68, 1, 2, 3, 4, 0x0F, 0xB6, 0x01, 0xC8,
            0x10, 0x00, 0x00,
        ];
        assert_eq!(lengths(false, code), [5, 3, 1, 7, 6, 5, 3, 4]);
        // VEX: vaddps ymm0, ymm1, ymm2; vpshufd xmm0, xmm1, 0x1B; vzeroupper
        assert_eq!(lengths(true, &[0xC5, 0xF4, 0x58, 0xC2, 0xC5, 0xF9, 0x70, 0xC1, 0x1B, 0xC5, 0xF8, 0x77]), [4, 5, 3]);
    }

    #[test]
    fn any_bytes_are_decoded_without_end() {
        let mut s = 99u64;
        for wide in [false, true] {
            let mut x = X86::new(wide);
            let (mut n, mut longest) = (0usize, 0u8);
            for i in 0..2_000_000usize {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let b = if i % 3 == 0 { [0x66, 0x0F, 0x48, 0xC4, 0x62, 0xF3][(s % 6) as usize] } else { s as u8 };
                for c in 0..CONTEXTS { std::hint::black_box(x.context(c, s as u32, i)); }
                if let Some(d) = x.update(b) {
                    n += 1;
                    longest = longest.max(d.len);
                }
            }
            assert!(n > 300_000 && longest <= 40, "{n} instructions, the longest {longest} bytes");
        }
    }
}
