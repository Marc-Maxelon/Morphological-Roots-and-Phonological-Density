from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SIGNATURE_RE = re.compile(r"^(\d+)-(\d+)-(\d+)$")
WORD_CANDIDATES = ("word", "wordform", "word_form", "lemma", "item")


def normalise_word(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
    )


def find_word_column(df: pd.DataFrame) -> str:
    lookup = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in WORD_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            return col
    raise ValueError(f"Could not identify a word column. Columns: {list(df.columns)}")


def locate_scope(raw_dir: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    candidates = list(raw_dir.rglob("wordsearch.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find wordsearch.csv under {raw_dir}. "
            "Place it in data/raw/ or pass --scope."
        )
    raise RuntimeError(
        "Found multiple wordsearch.csv files. Specify one with --scope:\n"
        + "\n".join(map(str, candidates))
    )


def locate_morpholex(raw_dir: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    candidates = [
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in {".xlsx", ".xlsm"} and "morph" in p.name.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        all_books = list(raw_dir.rglob("*.xlsx")) + list(raw_dir.rglob("*.xlsm"))
        if len(all_books) == 1:
            return all_books[0]
        raise FileNotFoundError(
            f"Could not identify the MorphoLex workbook under {raw_dir}. "
            "Place it in data/raw/ or pass --morpholex."
        )
    raise RuntimeError(
        "Found multiple possible MorphoLex workbooks. Specify one with --morpholex:\n"
        + "\n".join(map(str, candidates))
    )


def read_morpholex(path: Path):
    book = pd.ExcelFile(path)
    records = []
    source_rows = 0
    sheets_used = 0

    for sheet in book.sheet_names:
        match = SIGNATURE_RE.fullmatch(str(sheet).strip())
        if not match:
            continue
        n_prefix, n_root, n_suffix = map(int, match.groups())
        frame = pd.read_excel(path, sheet_name=sheet)
        if frame.empty:
            continue
        word_col = find_word_column(frame)
        words = normalise_word(frame[word_col])
        valid = words.dropna()
        source_rows += len(frame)
        sheets_used += 1
        for word in valid:
            records.append({
                "word_key": word,
                "n_prefix": n_prefix,
                "n_root": n_root,
                "n_suffix": n_suffix,
                "morpholex_signature": str(sheet),
            })

    if not records:
        raise ValueError(
            "No MorphoLex rows found. Expected worksheet names like '1-1-0' "
            "and a word column in each worksheet."
        )

    raw = pd.DataFrame(records)
    signatures_per_word = raw.groupby("word_key")["morpholex_signature"].nunique()
    multi_signature_words = int((signatures_per_word > 1).sum())

    resolved = []
    conflicts = []
    for word, group in raw.groupby("word_key", sort=False):
        counts = Counter(group["morpholex_signature"])
        max_count = max(counts.values())
        winners = sorted(sig for sig, count in counts.items() if count == max_count)

        if len(winners) > 1:
            conflicts.append({
                "word_key": word,
                "signatures": ";".join(sorted(counts)),
                "counts": ";".join(f"{k}:{counts[k]}" for k in sorted(counts)),
                "resolution": "excluded_tied_mode",
            })
            continue

        chosen = winners[0]
        row = group[group["morpholex_signature"] == chosen].iloc[0]
        resolved.append({
            "word_key": word,
            "n_prefix": int(row["n_prefix"]),
            "n_root": int(row["n_root"]),
            "n_suffix": int(row["n_suffix"]),
            "morpholex_signature": chosen,
        })

        if len(counts) > 1:
            conflicts.append({
                "word_key": word,
                "signatures": ";".join(sorted(counts)),
                "counts": ";".join(f"{k}:{counts[k]}" for k in sorted(counts)),
                "resolution": f"modal_signature_selected:{chosen}",
            })

    morph = pd.DataFrame(resolved)
    audit = {
        "morpholex_source_rows": int(source_rows),
        "morpholex_sheets_used": int(sheets_used),
        "morpholex_unique_word_forms": int(raw["word_key"].nunique()),
        "morpholex_words_with_multiple_signatures": multi_signature_words,
        "morpholex_words_excluded_due_to_tied_signatures": int(
            sum(x["resolution"] == "excluded_tied_mode" for x in conflicts)
        ),
        "morpholex_words_with_modal_signature_selected": int(
            sum(x["resolution"].startswith("modal_signature_selected") for x in conflicts)
        ),
        "morpholex_unique_resolved_words": int(len(morph)),
    }
    return morph, pd.DataFrame(conflicts), audit


def main():
    project_dir = ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path)
    parser.add_argument("--morpholex", type=Path)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project_dir / "data" / "raw"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=project_dir / "data" / "processed"
    )

    args = parser.parse_args()

    scope_path = locate_scope(args.raw_dir, args.scope)
    morph_path = locate_morpholex(args.raw_dir, args.morpholex)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    scope = pd.read_csv(scope_path, low_memory=False)
    word_col = find_word_column(scope)
    raw_rows = len(scope)
    scope["word_key"] = normalise_word(scope[word_col])
    scope = scope.dropna(subset=["word_key"]).copy()
    valid_word_rows = len(scope)
    duplicate_mask = scope["word_key"].duplicated(keep=False)
    duplicate_word_forms = int(scope.loc[duplicate_mask, "word_key"].nunique())
    scope = scope.drop_duplicates(subset="word_key", keep="first").copy()

    morph, conflicts, morph_audit = read_morpholex(morph_path)
    merged = scope.merge(morph, on="word_key", how="inner", validate="one_to_one")
    matched_words = len(merged)
    zero_root = int((merged["n_root"] < 1).sum())

    audit = {
        "scope_file": str(scope_path),
        "morpholex_file": str(morph_path),
        "scope_source_rows": int(raw_rows),
        "scope_rows_with_valid_word": int(valid_word_rows),
        "scope_unique_words": int(len(scope)),
        "scope_duplicate_word_forms": duplicate_word_forms,
        **morph_audit,
        "words_matched_across_datasets": int(matched_words),
        "words_with_zero_or_invalid_roots": zero_root,
        "words_after_root_filter": int((merged["n_root"] >= 1).sum()),
    }

    merged.to_csv(args.processed_dir / "merged_dataset.csv", index=False)

    analysis = merged.loc[merged["n_root"] >= 1].copy()
    required = [
        "PLD20", "NPhon", "n_prefix", "n_suffix", "n_root",
        "Freq_SUBTLEXUS_Zipf", "Valence_Warr",
    ]
    missing_cols = [c for c in required if c not in analysis.columns]
    if missing_cols:
        raise KeyError(
            f"SCOPE file is missing required columns: {missing_cols}. "
            f"Available columns: {list(analysis.columns)}"
        )

    analysis = analysis.dropna(subset=required).copy()
    analysis["multi_root"] = (analysis["n_root"] >= 2).astype(int)
    analysis.to_csv(args.processed_dir / "analysis_dataset.csv", index=False)

    audit["complete_analysis_observations"] = int(len(analysis))
    audit["complete_one_root_words"] = int((analysis["multi_root"] == 0).sum())
    audit["complete_multi_root_words"] = int((analysis["multi_root"] == 1).sum())
    audit["missing_values_after_matching_and_root_filter"] = {
        col: int(merged.loc[merged["n_root"] >= 1, col].isna().sum())
        for col in required
    }

    (args.processed_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    conflicts.to_csv(args.processed_dir / "morpholex_conflicts.csv", index=False)

    print("\n--- DATA PREPARATION COMPLETE ---")
    for key, value in audit.items():
        if not isinstance(value, dict):
            print(f"{key:48} {value}")
    print(f"\nMerged dataset:   {args.processed_dir / 'merged_dataset.csv'}")
    print(f"Analysis dataset: {args.processed_dir / 'analysis_dataset.csv'}")
    print(f"Audit:            {args.processed_dir / 'data_audit.json'}")


if __name__ == "__main__":
    main()
