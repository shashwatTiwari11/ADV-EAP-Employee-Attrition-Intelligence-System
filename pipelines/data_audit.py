import argparse
import pandas as pd


def audit_dataset(dataset_path: str, target_column: str):

    df = pd.read_csv(dataset_path)

    print("\n================ DATASET OVERVIEW ================\n")

    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print("\nColumn Names:\n")
    print(df.columns.tolist())

    if target_column not in df.columns:
        raise ValueError(f"\n❌ Target column '{target_column}' not found in dataset.")

    print("\n================ TARGET DISTRIBUTION ================\n")
    print(df[target_column].value_counts())
    print("\nClass Ratio (%):")
    print(df[target_column].value_counts(normalize=True) * 100)

    print("\n================ MISSING VALUES (%) ================\n")
    missing_percent = df.isnull().mean() * 100
    print(missing_percent[missing_percent > 0].sort_values(ascending=False))

    print("\n================ DATA TYPES ================\n")
    print(df.dtypes)

    print("\n================ DATE-LIKE COLUMNS (POTENTIAL LEAKAGE) ================\n")
    date_columns = [col for col in df.columns if "date" in col.lower()]
    print(date_columns if date_columns else "No obvious date columns found.")

    print("\n================ SAMPLE DATA ================\n")
    print(df.head())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audit Dataset")

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset CSV"
    )

    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Target column name"
    )

    args = parser.parse_args()

    audit_dataset(
        dataset_path=args.dataset_path,
        target_column=args.target_column
    )
