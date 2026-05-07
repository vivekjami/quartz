// quartz-core/src/wal.rs
use crc32fast::Hasher as Crc32Hasher;
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

pub const OP_INSERT: u8 = 0x01;
pub const OP_DELETE: u8 = 0x02;
pub const OP_CHECKPOINT: u8 = 0xFF;

/// Header: [entry_len: u32 LE][checksum: u32 LE]
const HEADER_SIZE: usize = 8;

pub struct WriteAheadLog {
    writer: BufWriter<File>,
    path: std::path::PathBuf,
}

impl WriteAheadLog {
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            path: path.to_path_buf(),
        })
    }

    /// Append a WAL entry. Panics if fsync fails (correct behavior: crash loudly).
    pub fn append(&mut self, op: u8, doc_id: u64, payload: &[u8]) -> std::io::Result<()> {
        let mut entry = Vec::with_capacity(9 + payload.len());
        entry.push(op);
        entry.extend_from_slice(&doc_id.to_le_bytes());
        entry.extend_from_slice(payload);

        let checksum = {
            let mut h = Crc32Hasher::new();
            h.update(&entry);
            h.finalize()
        };

        let header = {
            let mut h = [0u8; HEADER_SIZE];
            h[0..4].copy_from_slice(&(entry.len() as u32).to_le_bytes());
            h[4..8].copy_from_slice(&checksum.to_le_bytes());
            h
        };

        self.writer.write_all(&header)?;
        self.writer.write_all(&entry)?;
        self.writer.flush()?;
        // fsync: force OS buffer to disk. This is what makes WAL crash-safe.
        self.writer.get_ref().sync_data()?;
        Ok(())
    }

    pub fn checkpoint(&mut self, seg_id: u64) -> std::io::Result<()> {
        let payload = seg_id.to_le_bytes();
        self.append(OP_CHECKPOINT, 0, &payload)
    }

    /// Replay WAL entries since the last checkpoint.
    /// Returns Vec<(op, doc_id, payload)> for all entries after the final checkpoint.
    pub fn replay(path: &Path) -> std::io::Result<Vec<(u8, u64, Vec<u8>)>> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
            Err(e) => return Err(e),
        };
        let mut reader = BufReader::new(file);
        let mut all_entries: Vec<(u8, u64, Vec<u8>)> = Vec::new();
        let mut last_checkpoint_pos = 0usize;

        loop {
            let mut header = [0u8; HEADER_SIZE];
            match reader.read_exact(&mut header) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }

            let entry_len = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
            let expected_crc = u32::from_le_bytes(header[4..8].try_into().unwrap());

            let mut entry = vec![0u8; entry_len];
            match reader.read_exact(&mut entry) {
                Ok(_) => {}
                Err(_) => {
                    // Truncated entry: stop here (crash happened mid-write)
                    eprintln!("WAL: truncated entry detected, stopping replay");
                    break;
                }
            }

            // Verify checksum
            let actual_crc = {
                let mut h = Crc32Hasher::new();
                h.update(&entry);
                h.finalize()
            };
            if actual_crc != expected_crc {
                eprintln!("WAL: checksum mismatch, stopping replay (corruption detected)");
                break;
            }

            let op = entry[0];
            let doc_id = u64::from_le_bytes(entry[1..9].try_into().unwrap());
            let payload = entry[9..].to_vec();

            all_entries.push((op, doc_id, payload));

            if op == OP_CHECKPOINT {
                last_checkpoint_pos = all_entries.len(); // now points past the checkpoint
            }
        }

        // Return only entries after the last checkpoint
        Ok(all_entries[last_checkpoint_pos..].to_vec())
    }

    /// Returns the current byte offset in the WAL file.
    /// Useful for checkpointing: record this offset, replay only from here next time.
    pub fn current_offset(&mut self) -> std::io::Result<u64> {
        self.writer.flush()?;
        self.writer.get_mut().stream_position()
    }

    /// Replay this WAL using the stored path — convenience wrapper around replay().
    pub fn replay_self(&self) -> std::io::Result<Vec<(u8, u64, Vec<u8>)>> {
        Self::replay(&self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_append_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");

        let mut wal = WriteAheadLog::open(&path).unwrap();
        wal.append(OP_INSERT, 42, b"hello").unwrap();
        wal.append(OP_INSERT, 43, b"world").unwrap();
        drop(wal);

        let entries = WriteAheadLog::replay(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, OP_INSERT);
        assert_eq!(entries[0].1, 42);
        assert_eq!(entries[0].2, b"hello");
        assert_eq!(entries[1].1, 43);
        assert_eq!(entries[1].2, b"world");
    }

    #[test]
    fn checkpoint_truncates_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");

        let mut wal = WriteAheadLog::open(&path).unwrap();
        wal.append(OP_INSERT, 1, b"before checkpoint").unwrap();
        wal.checkpoint(99).unwrap();
        wal.append(OP_INSERT, 2, b"after checkpoint").unwrap();
        drop(wal);

        let entries = WriteAheadLog::replay(&path).unwrap();
        // Only entries AFTER the last checkpoint come back
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1, 2);
        assert_eq!(entries[0].2, b"after checkpoint");
    }

    #[test]
    fn replay_nonexistent_file_returns_empty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("does_not_exist.bin");
        let entries = WriteAheadLog::replay(&path).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn replay_self_matches_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");

        let mut wal = WriteAheadLog::open(&path).unwrap();
        wal.append(OP_INSERT, 7, b"test").unwrap();

        // replay_self uses self.path internally
        let entries = wal.replay_self().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].1, 7);
    }

    #[test]
    fn current_offset_advances_after_writes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("wal.bin");

        let mut wal = WriteAheadLog::open(&path).unwrap();
        let offset_before = wal.current_offset().unwrap();
        assert_eq!(offset_before, 0);

        wal.append(OP_INSERT, 1, b"payload").unwrap();
        let offset_after = wal.current_offset().unwrap();
        assert!(offset_after > offset_before, "offset must grow after write");
    }
}
