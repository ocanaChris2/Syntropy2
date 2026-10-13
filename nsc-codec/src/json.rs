//! Minimal JSON reader (std-only), enough for `tokenizer.json`: objects keep their key order,
//! numbers stay as their source text, and strings fully decode escapes (including surrogate pairs).

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<Value>),
    Obj(Vec<(String, Value)>),
}

impl Value {
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self { Value::Obj(v) => v.iter().find(|(k, _)| k == key).map(|(_, v)| v), _ => None }
    }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }
    pub fn as_arr(&self) -> Option<&[Value]> { if let Value::Arr(v) = self { Some(v) } else { None } }
    pub fn as_obj(&self) -> Option<&[(String, Value)]> { if let Value::Obj(v) = self { Some(v) } else { None } }
    pub fn as_u64(&self) -> Option<u64> { if let Value::Num(n) = self { n.parse().ok() } else { None } }
}

pub fn parse(text: &str) -> Result<Value, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0, depth: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() { return Err(format!("json: trailing data at byte {}", p.i)); }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn ws(&mut self) { while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') { self.i += 1; } }

    fn err<T>(&self, what: &str) -> Result<T, String> { Err(format!("json: {what} at byte {}", self.i)) }

    fn value(&mut self) -> Result<Value, String> {
        if self.depth > 256 { return self.err("nesting too deep"); }
        match self.b.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                self.depth += 1;
                let mut v = vec![];
                self.ws();
                if self.b.get(self.i) == Some(&b'}') { self.i += 1; self.depth -= 1; return Ok(Value::Obj(v)); }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') { return self.err("expected key"); }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') { return self.err("expected ':'"); }
                    self.i += 1;
                    self.ws();
                    let val = self.value()?;
                    v.push((k, val));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => { self.i += 1; self.depth -= 1; return Ok(Value::Obj(v)); }
                        _ => return self.err("expected ',' or '}'"),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                self.depth += 1;
                let mut v = vec![];
                self.ws();
                if self.b.get(self.i) == Some(&b']') { self.i += 1; self.depth -= 1; return Ok(Value::Arr(v)); }
                loop {
                    self.ws();
                    v.push(self.value()?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => { self.i += 1; self.depth -= 1; return Ok(Value::Arr(v)); }
                        _ => return self.err("expected ',' or ']'"),
                    }
                }
            }
            Some(b'"') => Ok(Value::Str(self.string()?)),
            Some(b't') if self.b[self.i..].starts_with(b"true") => { self.i += 4; Ok(Value::Bool(true)) }
            Some(b'f') if self.b[self.i..].starts_with(b"false") => { self.i += 5; Ok(Value::Bool(false)) }
            Some(b'n') if self.b[self.i..].starts_with(b"null") => { self.i += 4; Ok(Value::Null) }
            Some(c) if *c == b'-' || c.is_ascii_digit() => {
                let s = self.i;
                while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') { self.i += 1; }
                Ok(Value::Num(String::from_utf8_lossy(&self.b[s..self.i]).into_owned()))
            }
            _ => self.err("unexpected character"),
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        if self.i + 4 > self.b.len() { return self.err("truncated \\u escape"); }
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4]).map_err(|_| "json: bad \\u escape".to_string())?;
        let v = u32::from_str_radix(s, 16).map_err(|_| "json: bad \\u escape".to_string())?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' { self.i += 1; }
            out.push_str(std::str::from_utf8(&self.b[start..self.i]).map_err(|_| "json: invalid utf-8".to_string())?);
            match self.b.get(self.i) {
                Some(b'"') => { self.i += 1; return Ok(out); }
                Some(b'\\') => {
                    self.i += 1;
                    let c = *self.b.get(self.i).ok_or("json: truncated escape")?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'), b'\\' => out.push('\\'), b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'), b'f' => out.push('\u{c}'), b'n' => out.push('\n'),
                        b'r' => out.push('\r'), b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if self.b.get(self.i) != Some(&b'\\') || self.b.get(self.i + 1) != Some(&b'u') { return self.err("lone surrogate"); }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) { return self.err("bad surrogate pair"); }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else { hi };
                            out.push(char::from_u32(cp).ok_or("json: invalid code point")?);
                        }
                        _ => return self.err("unknown escape"),
                    }
                }
                _ => return self.err("unterminated string"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_and_escapes() {
        let v = parse(r#"{"a": [1, -2.5e3, true, null], "b": {"c": "xĠy\"\\😀"}}"#).unwrap();
        assert_eq!(v.get("a").unwrap().as_arr().unwrap().len(), 4);
        assert_eq!(v.get("b").unwrap().get("c").unwrap().as_str().unwrap(), "x\u{120}y\"\\\u{1F600}");
        assert!(parse("{\"a\": }").is_err());
        assert!(parse("[1, 2").is_err());
    }
}
