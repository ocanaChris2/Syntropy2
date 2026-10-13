//! Type 1 fonts: out of their two ciphers.
//!
//! A Type 1 font program — what LaTeX and many other writers embed in a PDF — keeps its private
//! dictionary behind `eexec`, a byte cipher with a fixed key, and inside that every glyph's
//! charstring behind the same cipher with another fixed key. Neither protects anything; both
//! make the font look like noise to a model. [`split`] finds the enciphered part of the font
//! at the start of a buffer and gives it in the clear, charstrings included; [`rebuild`]
//! enciphers it again. The cipher is a bijection byte for byte, so nothing else is kept.

/// The keys of `eexec` and of charstrings.
const EEXEC: u16 = 55665;
const CHARSTRING: u16 = 4330;
/// How far the `eexec` keyword is looked for, and the least enciphered bytes worth taking.
const HEAD: usize = 256 << 10;
const MIN: usize = 512;

fn decipher(b: &mut [u8], mut r: u16) {
    for x in b {
        let c = *x;
        *x = c ^ (r >> 8) as u8;
        r = (c as u16).wrapping_add(r).wrapping_mul(52845).wrapping_add(22719);
    }
}

fn encipher(b: &mut [u8], mut r: u16) {
    for x in b {
        let c = *x ^ (r >> 8) as u8;
        *x = c;
        r = (c as u16).wrapping_add(r).wrapping_mul(52845).wrapping_add(22719);
    }
}

/// Apply `f` to each charstring of the clear private dictionary `b`: the bytes a count and
/// `RD` or `-|` announce. The same charstrings are found whether they are enciphered or not.
fn charstrings(b: &mut [u8], f: impl Fn(&mut [u8], u16)) {
    let mut i = 0usize;
    while i + 4 <= b.len() {
        if b[i] == b' ' && b[i + 3] == b' ' && (&b[i + 1..i + 3] == b"RD" || &b[i + 1..i + 3] == b"-|") {
            // the count before it
            let mut d = i;
            while d > 0 && i - d < 7 && b[d - 1].is_ascii_digit() { d -= 1; }
            let n = b[d..i].iter().fold(0usize, |n, &x| n * 10 + (x - b'0') as usize);
            if d < i && d > 0 && b[d - 1].is_ascii_whitespace() && i + 4 + n <= b.len() {
                f(&mut b[i + 4..i + 4 + n], CHARSTRING);
                i += 4 + n;
                continue;
            }
        }
        i += 1;
    }
}

/// The enciphered part of the font at the start of `b`: (where it starts, its length); None
/// if `b` does not start with a Type 1 font whose end is in sight. `whole`: `b` is a stream
/// of its own (a PDF's font stream, inflated), so a font without the closing zeros — PDF
/// writers drop them — runs to its end.
pub fn extent(b: &[u8], whole: bool) -> Option<(usize, usize)> {
    if !(b.starts_with(b"%!PS-AdobeFont") || b.starts_with(b"%!FontType1")) { return None; }
    let head = &b[..b.len().min(HEAD)];
    let key = b"currentfile eexec";
    let mut start = head.windows(key.len()).position(|w| w == key)? + key.len();
    start += match b.get(start..start + 2)? {
        [b'\r', b'\n'] => 2,
        [x, _] if x.is_ascii_whitespace() => 1,
        _ => return None,
    };
    // in hexadecimal it is text already
    if b.get(start..start + 4)?.iter().all(|x| x.is_ascii_hexdigit()) { return None; }
    // it ends where the zeros before `cleartomark` begin
    let mark = b"cleartomark";
    let mut end = match b[start..].windows(mark.len()).position(|w| w == mark) {
        Some(p) => start + p,
        None if whole => b.len(),
        None => return None,
    };
    if end < b.len() { while end > start && matches!(b[end - 1], b'0' | b'\r' | b'\n') { end -= 1; } }
    (end - start >= MIN).then_some((start, end - start))
}

/// The enciphered part of the font at the start of `b` in the clear: (where it starts, the
/// clear bytes, as many as were enciphered).
pub fn split(b: &[u8], whole: bool) -> Option<(usize, Vec<u8>)> {
    let (start, len) = extent(b, whole)?;
    let mut plain = b[start..start + len].to_vec();
    decipher(&mut plain, EEXEC);
    // only what is a private dictionary: the keyword every one has
    if !plain.windows(8).any(|w| w == b"/Private") && !plain.windows(12).any(|w| w == b"/CharStrings") { return None; }
    charstrings(&mut plain, decipher);
    Some((start, plain))
}

/// Append the enciphered bytes of `plain` ([`split`]'s).
pub fn rebuild(plain: &[u8], out: &mut Vec<u8>) {
    let at = out.len();
    out.extend_from_slice(plain);
    charstrings(&mut out[at..], encipher);
    encipher(&mut out[at..], EEXEC);
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A font program of `glyphs` glyphs, as a PDF holds it.
    pub fn font(glyphs: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let mut rng = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let mut private = b"abcd dup /Private 8 dict dup begin /RD {string currentfile exch readstring pop} executeonly def /lenIV 4 def\n/CharStrings 9 dict dup begin\n".to_vec();
        for g in 0..glyphs {
            let n = 4 + (rng() % 60) as usize;
            let mut cs: Vec<u8> = (0..n).map(|k| if k < 4 { rng() as u8 } else { [139, 140, 5, 21, 22, 30, 31, 9][(rng() % 8) as usize] }).collect();
            encipher(&mut cs, CHARSTRING);
            private.extend_from_slice(format!("/glyph{g} {n} {} ", if g % 2 == 0 { "RD" } else { "-|" }).as_bytes());
            private.extend_from_slice(&cs);
            private.extend_from_slice(b" ND\n");
        }
        private.extend_from_slice(b"end end readonly put dup /FontName get exch definefont pop mark currentfile closefile\n");
        encipher(&mut private, EEXEC);
        let mut out = b"%!PS-AdobeFont-1.0: Test 001.000\n11 dict begin /FontType 1 def /FontName /Test def\ncurrentdict end\ncurrentfile eexec\n".to_vec();
        out.extend_from_slice(&private);
        for _ in 0..8 { out.extend_from_slice(&[b'0'; 64]); out.push(b'\n'); }
        out.extend_from_slice(b"cleartomark\n");
        out
    }

    #[test]
    fn fonts_come_out_of_their_ciphers_and_go_back() {
        for (glyphs, seed) in [(40usize, 3u64), (300, 4), (1, 5)] {
            let mut f = font(glyphs, seed);
            f.extend_from_slice(b"what follows");
            let Some((start, plain)) = split(&f, false) else { assert!(glyphs == 1, "{glyphs} glyphs: not split"); continue };
            assert!(plain.windows(12).any(|w| w == b"/CharStrings"));
            // the charstrings are in the clear: their operators, not noise
            let ops = plain.iter().filter(|x| [139u8, 140, 5, 21, 22, 30, 31, 9].contains(x)).count();
            assert!(ops * 3 > plain.len(), "{glyphs} glyphs: {ops} operator bytes of {}", plain.len());
            let mut back = vec![];
            rebuild(&plain, &mut back);
            assert!(back == f[start..start + plain.len()], "{glyphs} glyphs");
        }
        assert!(split(b"%!PS-AdobeFont-1.0: none\ncurrentfile eexec\nabc", true).is_none() && split(b"%PDF-1.4", true).is_none());
        // without the closing zeros, as a stream of its own
        let f = font(60, 8);
        let cut = &f[..f.len() - 8 * 65 - 12];
        assert!(split(cut, false).is_none());
        let (start, plain) = split(cut, true).unwrap();
        let mut back = vec![];
        rebuild(&plain, &mut back);
        assert!(start + plain.len() == cut.len() && back == cut[start..]);
        // any bytes go through the ciphers and back
        let mut s = 9u64;
        for _ in 0..300 {
            let n = (s % 3000) as usize;
            let b: Vec<u8> = (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; if s % 5 == 0 { b" 12 RD "[(s >> 8) as usize % 7] } else { s as u8 } }).collect();
            let mut plain = b.clone();
            decipher(&mut plain, EEXEC);
            charstrings(&mut plain, decipher);
            let mut back = vec![];
            rebuild(&plain, &mut back);
            assert!(back == b);
        }
    }
}
