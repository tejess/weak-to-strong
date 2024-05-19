import random
from random import shuffle
import csv

def get_hierarch_data(n_samples=1000, operation='sum', k=3, n=5):
    def sample_one():
        values = ['True', 'False']
        label = random.choice(values)
        if label == 'True':
            a, c = sample_equiv(operation=operation, k=k, n=n)
            b, d = sample_equiv(operation=operation, k=k, n=n)
        else:
            p = q = True
            while p and q:
                p, q = [random.choice([True, False]) for _ in range(2)]
            if p and not q:
                a, c = sample_equiv(operation=operation, k=k, n=n)
                b, d = sample_not_equiv(operation=operation, k=k, n=n)
            elif q and not p:
                a, c = sample_not_equiv(operation=operation, k=k, n=n)
                b, d = sample_equiv(operation=operation, k=k, n=n)
            else:
                a, c = sample_not_equiv(operation=operation, k=k, n=n)
                b, d = sample_not_equiv(operation=operation, k=k, n=n)
        return a, b, c, d, label
    
    sampled_data = [sample_one() for _ in range(n_samples)]
    sampled_input = ["({},{}) ({},{})".format(a, b, c, d) \
                     for a, b, c, d, _ in sampled_data]
    sampled_label = [label for _, _, _, _, label in sampled_data]
    return (sampled_input, sampled_label)

def sample_equiv(start=0, end=500, operation='sum', k=3, n=5):
    if operation == 'sum':
        b = random.randint(start, end-k)
        a = b + k
    elif operation == 'product':
        b = random.randint(start, end//k)
        a = b * k
    elif operation == 'composition':
        b = random.randint(start, end//k-n)
        a = b * k + n
    return a, b

def sample_not_equiv(start=0, end=99, operation='sum', k=3, n=5):
    a = random.randint(start, end)
    b = random.randint(start, end)
    if operation == 'sum':
        while a == b + k:
            b = random.randint(start, end)
    elif operation == 'product':
        while a == b * k:
            b = random.randint(start, end)
    elif operation == 'composition':
        while a == b * k + n:
            b = random.randint(start, end)
    return a, b

def main():
    n_samples = 4000
    operation = 'product'
    k = 5
    n = 5
    sampled_input, sampled_label = get_hierarch_data(n_samples=n_samples, operation=operation, k=k, n=n)

    # Combine samples and labels into a list of tuples and shuffle them
    data = list(zip(sampled_input, sampled_label))
    shuffle(data)

    # Split data by label to ensure the train dataset is balanced
    true_data = [d for d in data if d[1] == "True"]
    false_data = [d for d in data if d[1] == "False"]

    print(len(true_data))
    print(len(false_data))

    # Define number of samples per label for the train dataset
    samples_per_label_for_train = 1000

    # Create the train dataset with balanced True and False
    train_data = true_data[:samples_per_label_for_train] + false_data[:samples_per_label_for_train]
    print(len(train_data))
    shuffle(train_data)  # Shuffle to mix True and False labels

    # Use the remaining data for validation and test datasets
    remaining_data = true_data[samples_per_label_for_train:] + false_data[samples_per_label_for_train:]
    shuffle(remaining_data)  # Shuffle the remaining data

    # Split the remaining data into validation and test datasets
    split_index = len(remaining_data) // 2
    validation_data = remaining_data[:split_index]
    test_data = remaining_data[split_index:]

    # File paths
    train_file = '/data/tejess/weak-to-strong/train.csv'
    validation_file = '/data/tejess/weak-to-strong/validation.csv'
    test_file = '/data/tejess/weak-to-strong/test.csv'

    # Open all files simultaneously
    with open(train_file, 'w', newline='') as f_train, \
        open(validation_file, 'w', newline='') as f_val, \
        open(test_file, 'w', newline='') as f_test:
        
        # Create CSV Writers for each file
        writer_train = csv.writer(f_train)
        writer_val = csv.writer(f_val)
        writer_test = csv.writer(f_test)
        
        # Write data to each file
        for item in train_data:
            writer_train.writerow(item)
        for item in validation_data:
            writer_val.writerow(item)
        for item in test_data:
            writer_test.writerow(item)


if __name__ == '__main__':
    main()