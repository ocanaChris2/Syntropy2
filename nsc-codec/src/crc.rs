//! CRC-32 (IEEE 802.3, the zlib/PNG/gzip polynomial), slicing-by-8 for ~2 GB/s.

const POLY: u32 = 0xEDB8_8320;

const fn build() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { POLY ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut i = 0;
    while i < 256 {
        let mut s = 1;
        while s < 8 {
            let prev = t[s - 1][i];
            t[s][i] = (prev >> 8) ^ t[0][(prev & 0xff) as usize];
            s += 1;
        }
        i += 1;
    }
    t
}

static T: [[u32; 256]; 8] = build();

/// Incremental CRC-32.
#[derive(Clone, Copy)]
pub struct Crc32(u32);

impl Crc32 {
    pub fn new() -> Self { Crc32(!0) }

    pub fn update(&mut self, data: &[u8]) {
        let mut c = self.0;
        let mut chunks = data.chunks_exact(8);
        for ch in &mut chunks {
            let lo = c ^ u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]);
            let hi = u32::from_le_bytes([ch[4], ch[5], ch[6], ch[7]]);
            c = T[7][(lo & 0xff) as usize] ^ T[6][((lo >> 8) & 0xff) as usize]
                ^ T[5][((lo >> 16) & 0xff) as usize] ^ T[4][(lo >> 24) as usize]
                ^ T[3][(hi & 0xff) as usize] ^ T[2][((hi >> 8) & 0xff) as usize]
                ^ T[1][((hi >> 16) & 0xff) as usize] ^ T[0][(hi >> 24) as usize];
        }
        for &b in chunks.remainder() {
            c = (c >> 8) ^ T[0][((c ^ b as u32) & 0xff) as usize];
        }
        self.0 = c;
    }

    pub fn finish(self) -> u32 { !self.0 }
}

/// One-shot CRC-32 of `data`.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b"The quick brown fox jumps over the lazy dog"), 0x414F_A339);
        // incremental == one-shot, across odd split points
        let data: Vec<u8> = (0..10_000u32).map(|i| (i * 2654435761u32 >> 13) as u8).collect();
        let mut c = Crc32::new();
        c.update(&data[..3]);
        c.update(&data[3..1001]);
        c.update(&data[1001..]);
        assert_eq!(c.finish(), crc32(&data));
    }
}
