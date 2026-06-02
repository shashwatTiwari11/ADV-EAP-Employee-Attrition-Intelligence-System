import argparse
from src.training.trainer import ModelTrainer


def run_training_pipeline(features_path: str, target_column: str, model_type: str):
    """
    Runs the full training pipeline.
    """

    trainer = ModelTrainer(

        target_column=target_column,
        model_type=model_type
        
    )

    results = trainer.train(features_path)

    print("\n✅ MODEL TRAINING COMPLETE")
    print(f"Model Type: {model_type}")
    print(f"Accuracy: {results['accuracy']}")
    print(f"ROC-AUC: {results['roc_auc']}")
    print(f"Precision: {results['precision']}")
    print(f"Recall: {results['recall']}")
    print(f"F1-Score: {results['f1_score']}")
    # print(f"Registered Model Version: {results['model_version']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)

    args = parser.parse_args()

    run_training_pipeline(
        features_path=args.features_path,
        target_column=args.target_column,
        model_type=args.model_type
    )
