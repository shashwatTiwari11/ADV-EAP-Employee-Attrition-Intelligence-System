import argparse
from src.features.feature_builder import FeatureBuilder


def run_feature_pipeline(reference_csv_path: str, target_column: str):

    drop_columns = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ]

    builder = FeatureBuilder(
        target_column=target_column,
        drop_columns=drop_columns
    )

    output_path = builder.build(reference_csv_path)

    print(f"\n✅ Features created at: {output_path}")

    return output_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Feature Pipeline")

    parser.add_argument(
        "--reference_path",
        type=str,
        required=True,
        help="Path to reference dataset"
    )

    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Target column name"
    )

    args = parser.parse_args()

    run_feature_pipeline(
        reference_csv_path=args.reference_path,
        target_column=args.target_column
    )
