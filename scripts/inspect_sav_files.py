from pathlib import Path
import pyreadstat

raw_extracted_dir = Path("data/raw/extracted")

sav_files = list(raw_extracted_dir.rglob("*.sav"))

print(f"Number of .sav files found: {len(sav_files)}")
print("-" * 80)

for sav_path in sav_files:
    print(f"\nFILE: {sav_path}")

    try:
        df, meta = pyreadstat.read_sav(
            sav_path,
            metadataonly=True
        )

        print(f"Rows: {meta.number_rows}")
        print(f"Columns: {len(meta.column_names)}")
        print("First 30 variables:")
        print(meta.column_names[:30])

    except Exception as e:
        print(f"ERROR reading file: {e}")

 


  