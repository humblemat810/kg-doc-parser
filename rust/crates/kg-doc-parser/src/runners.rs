use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn walk_directory(
    directory: &Path,
    allowed_suffixes: &std::collections::HashSet<String>,
    recursive: bool,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && recursive {
            walk_directory(&path, allowed_suffixes, recursive, files)?;
        } else if path.is_file() {
            let suffix = path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .map(|value| format!(".{}", value.to_lowercase()))
                .unwrap_or_default();
            if allowed_suffixes.contains(&suffix) {
                files.push(path);
            }
        }
    }
    Ok(())
}

pub fn discover_input_files(
    paths: &[PathBuf],
    allowed_suffixes: &[&str],
    recursive: bool,
) -> Result<Vec<PathBuf>> {
    let allowed_suffixes = allowed_suffixes
        .iter()
        .map(|suffix| suffix.to_lowercase())
        .collect::<std::collections::HashSet<_>>();
    let mut files = Vec::new();
    for raw_path in paths {
        let path = raw_path.as_path();
        if path.is_dir() {
            walk_directory(path, &allowed_suffixes, recursive, &mut files)?;
        } else {
            let suffix = path
                .extension()
                .and_then(std::ffi::OsStr::to_str)
                .map(|value| format!(".{}", value.to_lowercase()))
                .unwrap_or_default();
            if allowed_suffixes.contains(&suffix) {
                files.push(path.to_path_buf());
            }
        }
    }
    let mut unique = BTreeMap::new();
    for path in files {
        let key = path.canonicalize().unwrap_or_else(|_| path.clone());
        unique.insert(key, path);
    }
    let mut result = unique.into_values().collect::<Vec<_>>();
    result.sort_by_key(|path| path.to_string_lossy().to_string());
    Ok(result)
}

pub fn collision_safe_batch_identity(
    source_path: &Path,
    stem_counts: &BTreeMap<String, usize>,
) -> String {
    let stem = source_path
        .file_stem()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or_default()
        .to_string();
    if stem_counts.get(&stem).copied().unwrap_or(0) <= 1 {
        return stem;
    }
    let resolved = source_path
        .canonicalize()
        .unwrap_or_else(|_| source_path.to_path_buf());
    let mut digest = Sha256::new();
    digest.update(resolved.to_string_lossy().as_bytes());
    let fingerprint = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("{stem}-{}", &fingerprint[..16])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn discovers_and_deduplicates_input_files() {
        let temp = tempfile::tempdir().unwrap();
        let nested = temp.path().join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        let alpha = temp.path().join("alpha.md");
        let beta = nested.join("beta.txt");
        std::fs::write(&alpha, "alpha").unwrap();
        std::fs::write(&beta, "beta").unwrap();
        let files = discover_input_files(
            &[temp.path().to_path_buf(), alpha.clone()],
            &[".md", ".txt"],
            true,
        )
        .unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.contains(&alpha));
        assert!(files.contains(&beta));
        let shallow =
            discover_input_files(&[temp.path().to_path_buf()], &[".md", ".txt"], false).unwrap();
        assert_eq!(shallow, vec![alpha]);
    }

    #[test]
    fn preserves_unique_stems_and_hashes_collisions() {
        let unique = collision_safe_batch_identity(
            Path::new("inputs/unique.md"),
            &BTreeMap::from([("unique".to_string(), 1)]),
        );
        assert_eq!(unique, "unique");
        let identity = collision_safe_batch_identity(
            Path::new("inputs/one/duplicate.md"),
            &BTreeMap::from([("duplicate".to_string(), 2)]),
        );
        assert_ne!(identity, "duplicate");
        assert!(identity.starts_with("duplicate-"));
        assert_eq!(identity.len(), "duplicate-".len() + 16);
    }
}
