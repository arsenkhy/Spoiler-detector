from sklearn.metrics import classification_report
def generate_reference_report():
    print("Predicting on test dataset...")
    # Read the contents of reference.txt
    with open('./data/reference/test_reference.txt', "r") as f:
        reference_labels = [int(line.strip()) for line in f.readlines()]

    # Read the contents of model_output.out
    with open('./output/model_output.out', "r") as f:
        model_output = [int(line.strip()) for line in f.readlines()]

    # Compute classification report
    report = classification_report(reference_labels, model_output, target_names=['not spoiler', 'spoiler'])
    print(report)

if __name__ == "__main__":
    report = generate_reference_report()
