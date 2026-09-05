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
use std::collections::HashMap;
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

    /// Every section record in slot order, which is the order of the rows in
    /// the vector file: a record's position here is its row there.
    pub fn sections(&self) -> Result<Vec<Section>> {
        let mut q = self.conn.prepare(
            "SELECT path, start, stop, heading, crumbs, fm, truncated
             FROM sections ORDER BY slot",
        )?;
        let mut out = Vec::new();
        let mut rows = q.query([])?;
        while let Some(r) = rows.next()? {
            out.push(Section {
                path: r.get(0)?,
                start: r.get::<_, i64>(1)? as usize,
                end: r.get::<_, i64>(2)? as usize,
                heading: r.get(3)?,
                breadcrumb: serde_json::from_str(&r.get::<_, String>(4)?)?,
                fm: serde_json::from_str::<Map<String, Value>>(&r.get::<_, String>(5)?)?,
                truncated: r.get::<_, i64>(6)? != 0,
                text: String::new(),
            });
        }
        Ok(out)
    }

    /// Replace the whole record set, in one transaction. The vector file is
    /// written by the caller before this is called: a crash between the two
    /// leaves bytes nothing points at, where the other order would leave a
    /// record pointing at a row that is not there.
    pub fn replace(
        &mut self,
        meta: &Meta,
        files: &HashMap<String, Stamp>,
        sections: &[Section],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM sections", [])?;
        tx.execute("DELETE FROM files", [])?;
        {
            let mut ins = tx.prepare(
                "INSERT INTO sections(slot, path, start, stop, heading, crumbs, fm, truncated)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            )?;
            for (slot, s) in sections.iter().enumerate() {
                ins.execute(params![
                    slot as i64,
                    s.path,
                    s.start as i64,
                    s.end as i64,
                    s.heading,
                    serde_json::to_string(&s.breadcrumb)?,
                    serde_json::to_string(&s.fm)?,
                    i64::from(s.truncated),
                ])?;
            }
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
        st.replace(&meta, &files, &[section("a.md", 1), section("a.md", 9)])
            .unwrap();

        let got = st.sections().unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].start, 1, "slot order is the order the rows went in");
        assert_eq!(got[1].start, 9);
        assert_eq!(got[0].fm["tags"], json!(["a", "b"]));
        assert!(got[0].truncated);
        assert_eq!(got[0].breadcrumb, vec!["Top".to_string()]);
        assert!(got[0].text.is_empty(), "the store holds no prose to return");

        // u64 hashes past i64::MAX and negative clocks both survive the trip.
        assert_eq!(st.files().unwrap()["a.md"].hash, u64::MAX);
        assert_eq!(st.files().unwrap()["a.md"].mtime, -1);
        assert_eq!(st.meta().unwrap(), meta);
    }

    #[test]
    fn replace_leaves_nothing_of_the_previous_set() {
        let dir = tempdir();
        let mut st = Store::open(&dir).unwrap();
        let meta = Meta { model: "m".into(), endpoint: "e".into(), dim: 3 };
        st.replace(&meta, &HashMap::new(), &[section("a.md", 1)])
            .unwrap();
        st.replace(&meta, &HashMap::new(), &[section("b.md", 1)])
            .unwrap();
        let got = st.sections().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].path, "b.md");
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
