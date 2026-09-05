//! The index's record store: section references, file stamps, and the metadata
//! naming the vector space. The vector matrix is not here — it sits beside this
//! file and is mapped, because a scan that touches every row is charged per-row
//! overhead by a B-tree and none by a flat file.
//!
//! Every statement folio issues is in this module. That is deliberate: what
//! folio asks of SQL is small enough to keep in one place, and an engine behind
//! it can be replaced without the rest of folio knowing.

use crate::sections::Section;
use anyhow::{Context, Result};
use rusqlite::{Connection, params};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

pub const DIR: &str = ".folio";

/// What a file looked like at index time: the content hash that decides whether
/// it is re-embedded, and the length and modification time that decide whether
/// it is read at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stamp {
    pub hash: u64,
    pub len: u64,
    pub mtime: i64,
}

/// The vector space the records belong to.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Meta {
    pub model: String,
    pub endpoint: String,
    pub dim: usize,
}

/// What the index covers.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub files: usize,
    pub sections: usize,
    pub truncated: usize,
}

pub struct Store {
    conn: Connection,
}

pub fn db_path(root: &Path) -> std::path::PathBuf {
    root.join(DIR).join("index.db")
}

pub fn vectors_path(root: &Path) -> std::path::PathBuf {
    root.join(DIR).join("vectors.f32")
}

impl Store {
    /// Open the store under `root`, creating it if it is not there.
    ///
    /// The journal mode is left at the default. WAL doubles the writes and
    /// leaves a sidecar as large as the database to buy concurrency that a
    /// short-lived single-process command never uses, and it does not work on a
    /// network filesystem, which a tool pointed at any directory has to.
    pub fn open(root: &Path) -> Result<Store> {
        let dir = root.join(DIR);
        fs::create_dir_all(&dir)?;
        // An index written by an older folio kept its records in these. They
        // are derived data, and leaving them is leaving a second answer to the
        // question this store now answers.
        for legacy in ["sections.jsonl", "state.json"] {
            let _ = fs::remove_file(dir.join(legacy));
        }
        let path = db_path(root);
        let conn = Connection::open(&path)
            .with_context(|| format!("{} could not be opened", path.display()))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS files(
                 path  TEXT PRIMARY KEY,
                 hash  INTEGER NOT NULL,
                 len   INTEGER NOT NULL,
                 mtime INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS sections(
                 slot      INTEGER PRIMARY KEY,
                 path      TEXT NOT NULL,
                 start     INTEGER NOT NULL,
                 stop      INTEGER NOT NULL,
                 heading   TEXT,
                 crumbs    TEXT NOT NULL,
                 fm        TEXT NOT NULL,
                 truncated INTEGER NOT NULL);
             CREATE INDEX IF NOT EXISTS sections_path ON sections(path);",
        )?;
        Ok(Store { conn })
    }

    pub fn meta(&self) -> Result<Meta> {
        let mut q = self.conn.prepare("SELECT k, v FROM meta")?;
        let mut m = Meta::default();
        let mut rows = q.query([])?;
        while let Some(r) = rows.next()? {
            let (k, v): (String, String) = (r.get(0)?, r.get(1)?);
            match k.as_str() {
                "model" => m.model = v,
                "endpoint" => m.endpoint = v,
                "dim" => m.dim = v.parse().unwrap_or(0),
                _ => {}
            }
        }
        Ok(m)
    }

    /// Set while the matrix is being compacted, because the rewritten matrix
    /// and the renumbered records land in two steps and a crash between them
    /// leaves records naming the wrong rows. Cleared by `renumber`.
    pub fn compacting(&self) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM meta WHERE k = 'compacting'",
            [],
            |r| r.get(0),
        )?;
        Ok(n > 0)
    }

    pub fn start_compacting(&self) -> Result<()> {
        self.conn
            .execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('compacting', '1')", [])?;
        Ok(())
    }

    pub fn files(&self) -> Result<HashMap<String, Stamp>> {
        let mut q = self.conn.prepare("SELECT path, hash, len, mtime FROM files")?;
        let mut out = HashMap::new();
        let mut rows = q.query([])?;
        while let Some(r) = rows.next()? {
            out.insert(
                r.get::<_, String>(0)?,
                Stamp {
                    hash: r.get::<_, i64>(1)? as u64,
                    len: r.get::<_, i64>(2)? as u64,
                    mtime: r.get(3)?,
                },
            );
        }
        Ok(out)
    }

    /// The highest slot any record names, plus one. Records are deleted
    /// without their rows being moved, so this is not the record count: it is
    /// the row the matrix has grown to and the slot the next append takes.
    pub fn high_water(&self) -> Result<usize> {
        let max: Option<i64> = self
            .conn
            .query_row("SELECT max(slot) FROM sections", [], |r| r.get(0))?;
        Ok(max.map_or(0, |m| m as usize + 1))
    }

    /// How many rows a query has to consider, without deserialising any of
    /// them. This is the whole cost of a query that carries no filter: ranking
    /// needs the vectors, and the records only become interesting once the top
    /// of the ranking is known.
    pub fn slots(&self) -> Result<Vec<usize>> {
        let mut q = self.conn.prepare("SELECT slot FROM sections ORDER BY slot")?;
        let mut out = Vec::new();
        let mut rows = q.query([])?;
        while let Some(r) = rows.next()? {
            out.push(r.get::<_, i64>(0)? as usize);
        }
        Ok(out)
    }

    /// Slots with their frontmatter, for a query that filters or joins. The
    /// heading trail and the line range are left on disk: nothing decides a
    /// query on those, they are only printed for the rows that win.
    pub fn slots_with_fm(&self) -> Result<Vec<(usize, Map<String, Value>)>> {
        let mut q = self.conn.prepare("SELECT slot, fm FROM sections ORDER BY slot")?;
        let mut out = Vec::new();
        let mut rows = q.query([])?;
        while let Some(r) = rows.next()? {
            out.push((
                r.get::<_, i64>(0)? as usize,
                serde_json::from_str(&r.get::<_, String>(1)?)?,
            ));
        }
        Ok(out)
    }

    /// The full records for the slots a query settled on, in the order asked.
    pub fn hydrate(&self, slots: &[usize]) -> Result<Vec<Section>> {
        let mut q = self.conn.prepare(
            "SELECT path, start, stop, heading, crumbs, fm, truncated
             FROM sections WHERE slot = ?1",
        )?;
        let mut out = Vec::new();
        for slot in slots {
            let s = q.query_row(params![*slot as i64], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, Option<String>>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                    r.get::<_, i64>(6)?,
                ))
            })?;
            out.push(Section {
                path: s.0,
                start: s.1 as usize,
                end: s.2 as usize,
                heading: s.3,
                breadcrumb: serde_json::from_str(&s.4)?,
                fm: serde_json::from_str(&s.5)?,
                truncated: s.6 != 0,
                text: String::new(),
            });
        }
        Ok(out)
    }

    /// What the index covers, counted in the database rather than by walking
    /// the records out of it.
    pub fn counts(&self) -> Result<Counts> {
        let (files, sections, truncated) = self.conn.query_row(
            "SELECT count(DISTINCT path), count(*), coalesce(sum(truncated), 0) FROM sections",
            [],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)),
        )?;
        Ok(Counts {
            files: files as usize,
            sections: sections as usize,
            truncated: truncated as usize,
        })
    }

    /// Retire the records of `dropped` and record `fresh` at the slots it was
    /// written to, in one transaction.
    ///
    /// The vector file is written by the caller before this is called: a crash
    /// between the two leaves rows at the end of the matrix that no record
    /// names, where the other order would leave records naming rows that were
    /// never written.
    ///
    /// A dropped record's row is not moved and not reused. Moving it would make
    /// the write proportional to the index rather than to the change, which is
    /// the whole reason the rows are appended.
    pub fn apply(
        &mut self,
        meta: &Meta,
        files: &HashMap<String, Stamp>,
        dropped: &HashSet<String>,
        fresh: &[(usize, Section)],
    ) -> Result<usize> {
        let mut retired = 0;
        let tx = self.conn.transaction()?;
        {
            let mut del = tx.prepare("DELETE FROM sections WHERE path = ?1")?;
            for path in dropped {
                retired += del.execute(params![path])?;
            }
            let mut ins = tx.prepare(
                "INSERT INTO sections(slot, path, start, stop, heading, crumbs, fm, truncated)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            )?;
            for (slot, s) in fresh {
                ins.execute(params![
                    *slot as i64,
                    s.path,
                    s.start as i64,
                    s.end as i64,
                    s.heading,
                    serde_json::to_string(&s.breadcrumb)?,
                    serde_json::to_string(&s.fm)?,
                    i64::from(s.truncated),
                ])?;
            }
            tx.execute("DELETE FROM files", [])?;
            let mut ins =
                tx.prepare("INSERT INTO files(path, hash, len, mtime) VALUES (?1,?2,?3,?4)")?;
            for (path, st) in files {
                ins.execute(params![path, st.hash as i64, st.len as i64, st.mtime])?;
            }
            let mut ins = tx.prepare("INSERT OR REPLACE INTO meta(k, v) VALUES (?1, ?2)")?;
            ins.execute(params!["model", meta.model])?;
            ins.execute(params!["endpoint", meta.endpoint])?;
            ins.execute(params!["dim", meta.dim.to_string()])?;
        }
        tx.commit()?;
        Ok(retired)
    }

    /// Drop every record, for a rebuild or a change of vector space. The
    /// caller truncates the matrix to match.
    pub fn clear(&self) -> Result<()> {
        self.conn.execute_batch("DELETE FROM sections; DELETE FROM files;")?;
        Ok(())
    }

    /// Renumber `kept` to slots 0.. in the order given, which is the order the
    /// caller wrote their rows into the new matrix.
    pub fn renumber(&mut self, kept: &[usize]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            // Through the negatives, because the target of one move is the
            // source of another and slot is the primary key.
            let mut mv = tx.prepare("UPDATE sections SET slot = ?2 WHERE slot = ?1")?;
            for (to, from) in kept.iter().enumerate() {
                mv.execute(params![*from as i64, -(to as i64) - 1])?;
            }
            tx.execute("UPDATE sections SET slot = -slot - 1 WHERE slot < 0", [])?;
            tx.execute("DELETE FROM meta WHERE k = 'compacting'", [])?;
        }
        tx.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn section(path: &str, start: usize) -> Section {
        Section {
            path: path.into(),
            start,
            end: start + 2,
            heading: Some("H".into()),
            breadcrumb: vec!["Top".into()],
            fm: json!({"id": "M-1", "tags": ["a", "b"]})
                .as_object()
                .unwrap()
                .clone(),
            truncated: true,
            text: "prose that must not survive a round trip".into(),
        }
    }

    #[test]
    fn a_record_round_trips_without_its_prose() {
        let dir = tempdir();
        let mut st = Store::open(&dir).unwrap();
        let meta = Meta { model: "m".into(), endpoint: "e".into(), dim: 3 };
        let files = HashMap::from([(
            "a.md".to_string(),
            Stamp { hash: u64::MAX, len: 12, mtime: -1 },
        )]);
        st.apply(&meta, &files, &HashSet::new(), &[(0, section("a.md", 1)), (7, section("a.md", 9))])
            .unwrap();

        assert_eq!(st.slots().unwrap(), vec![0, 7], "the slot names the matrix row");
        assert_eq!(st.high_water().unwrap(), 8, "the next append goes past the last row");
        assert_eq!(
            st.counts().unwrap(),
            Counts { files: 1, sections: 2, truncated: 2 }
        );

        // Read back in the order asked for, not the order stored.
        let got = st.hydrate(&[7, 0]).unwrap();
        assert_eq!((got[0].start, got[1].start), (9, 1));
        assert_eq!(got[1].fm["tags"], json!(["a", "b"]));
        assert!(got[1].truncated);
        assert_eq!(got[1].breadcrumb, vec!["Top".to_string()]);
        assert!(got[1].text.is_empty(), "the store holds no prose to return");

        // A filtering query reads frontmatter and nothing else.
        let fm = st.slots_with_fm().unwrap();
        assert_eq!(fm.iter().map(|(s, _)| *s).collect::<Vec<_>>(), vec![0, 7]);
        assert_eq!(fm[0].1["id"], json!("M-1"));

        // u64 hashes past i64::MAX and negative clocks both survive the trip.
        assert_eq!(st.files().unwrap()["a.md"].hash, u64::MAX);
        assert_eq!(st.files().unwrap()["a.md"].mtime, -1);
        assert_eq!(st.meta().unwrap(), meta);
    }

    #[test]
    fn a_retired_record_leaves_its_row_behind() {
        let dir = tempdir();
        let mut st = Store::open(&dir).unwrap();
        let meta = Meta { model: "m".into(), endpoint: "e".into(), dim: 3 };
        let none = HashSet::new();
        st.apply(&meta, &HashMap::new(), &none,
                 &[(0, section("a.md", 1)), (1, section("b.md", 1))]).unwrap();
        // a.md is re-indexed: its old row is not reused and not moved.
        st.apply(&meta, &HashMap::new(), &HashSet::from(["a.md".to_string()]),
                 &[(2, section("a.md", 5))]).unwrap();

        let slots = st.slots().unwrap();
        let paths: Vec<String> = st.hydrate(&slots).unwrap().into_iter().map(|s| s.path).collect();
        assert_eq!(slots, vec![1, 2], "slot 0 is dead space and b.md did not move");
        assert_eq!(paths, vec!["b.md".to_string(), "a.md".to_string()]);
        assert_eq!(st.high_water().unwrap(), 3);
    }

    #[test]
    fn compaction_renumbers_the_survivors_and_clears_its_flag() {
        let dir = tempdir();
        let mut st = Store::open(&dir).unwrap();
        let meta = Meta { model: "m".into(), endpoint: "e".into(), dim: 3 };
        st.apply(&meta, &HashMap::new(), &HashSet::new(),
                 &[(1, section("b.md", 1)), (4, section("c.md", 1)), (9, section("d.md", 1))])
            .unwrap();
        st.start_compacting().unwrap();
        assert!(st.compacting().unwrap());

        let kept = st.slots().unwrap();
        assert_eq!(kept, vec![1, 4, 9]);
        st.renumber(&kept).unwrap();

        let slots = st.slots().unwrap();
        let paths: Vec<String> = st.hydrate(&slots).unwrap().into_iter().map(|s| s.path).collect();
        assert_eq!(slots, vec![0, 1, 2], "survivors take the rows from the front");
        assert_eq!(
            paths,
            vec!["b.md".to_string(), "c.md".to_string(), "d.md".to_string()],
            "and keep their order"
        );
        assert!(!st.compacting().unwrap(), "the flag is cleared with the renumbering");
    }

    fn tempdir() -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "folio-store-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }
}
