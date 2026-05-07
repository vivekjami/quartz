// quartz-core/src/codec.rs

/// VByte encode a slice of unsigned integers.
/// Each integer is written as 1–5 bytes; the MSB of each byte is 1 if more bytes follow.
pub fn vbyte_encode(values: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        vbyte_encode_one(v, &mut out);
    }
    out
}

#[inline(always)]
pub fn vbyte_encode_one(mut v: u32, out: &mut Vec<u8>) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte); // MSB=0: last byte
            break;
        } else {
            out.push(byte | 0x80); // MSB=1: more bytes follow
        }
    }
}

/// Decode a VByte sequence into a Vec<u32>.
pub fn vbyte_decode(data: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let (val, consumed) = vbyte_decode_one(&data[i..]);
        out.push(val);
        i += consumed;
    }
    out
}

#[inline(always)]
pub fn vbyte_decode_one(data: &[u8]) -> (u32, usize) {
    let mut val: u32 = 0;
    let mut shift = 0;
    let mut i = 0;
    loop {
        let byte = data[i];
        i += 1;
        val |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (val, i)
}

/// Delta-encode a sorted postings list, then VByte encode the gaps.
/// Input must be sorted ascending.
pub fn encode_postings(doc_ids: &[u32], term_freqs: &[u8]) -> Vec<u8> {
    assert_eq!(doc_ids.len(), term_freqs.len());
    let mut out = Vec::new();
    vbyte_encode_one(doc_ids.len() as u32, &mut out); // doc_freq first
    let mut prev = 0u32;
    for (&doc_id, &tf) in doc_ids.iter().zip(term_freqs.iter()) {
        let gap = doc_id - prev;
        prev = doc_id;
        vbyte_encode_one(gap, &mut out);
        out.push(tf); // TF stored as raw byte (max 255, sufficient for web docs)
    }
    out
}

/// Decode a postings entry, returning (doc_ids, term_freqs).
pub fn decode_postings(data: &[u8]) -> (Vec<u32>, Vec<u8>) {
    let (doc_freq, mut offset) = vbyte_decode_one(data);
    let mut doc_ids = Vec::with_capacity(doc_freq as usize);
    let mut tfs = Vec::with_capacity(doc_freq as usize);
    let mut prev = 0u32;
    for _ in 0..doc_freq {
        let (gap, consumed) = vbyte_decode_one(&data[offset..]);
        offset += consumed;
        let doc_id = prev + gap;
        prev = doc_id;
        doc_ids.push(doc_id);
        tfs.push(data[offset]);
        offset += 1;
    }
    (doc_ids, tfs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_small() {
        let ids = vec![1u32, 5, 100, 10000, 1_000_000];
        let tfs = vec![1u8, 3, 1, 7, 2];
        let encoded = encode_postings(&ids, &tfs);
        let (decoded_ids, decoded_tfs) = decode_postings(&encoded);
        assert_eq!(decoded_ids, ids);
        assert_eq!(decoded_tfs, tfs);
    }

    #[test]
    fn single_byte_gap() {
        // Gaps of 1 should encode to 1 byte each
        let ids: Vec<u32> = (0..128).collect();
        let tfs = vec![1u8; 128];
        let encoded = encode_postings(&ids, &tfs);
        // doc_freq VByte + 128 * (1 byte gap + 1 byte tf)
        // doc_freq=128 needs 2 VByte bytes; each gap (0 or 1) is 1 byte
        assert!(encoded.len() < 300);
        let (d, t) = decode_postings(&encoded);
        assert_eq!(d, ids);
        assert_eq!(t, tfs);
    }

    #[test]
    fn large_doc_ids() {
        let ids = vec![0u32, 1_000_000, 2_000_000, u32::MAX - 1];
        let tfs = vec![1u8; 4];
        let (d, _) = decode_postings(&encode_postings(&ids, &tfs));
        assert_eq!(d, ids);
    }
}
