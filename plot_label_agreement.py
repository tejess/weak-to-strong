import json
import matplotlib.pyplot as plt
from datasets import load_from_disk
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--strong_labels_path', 
                    help="""Path to the strong labels predicted during training.
                    Should be a JSON file.""")
parser.add_argument('--weak_labels_path',
                    help="""Path to the labels predicted by the weak model.
                    Should be a directory ending with weak_labels/""")
parser.add_argument('--save_path', help='Path to the directory to save plots')
parser.add_argument('--batch_size', help='Batch size used to train the strong model', required=False)


# Parse arguments from command line
args = parser.parse_args()

with open(args.strong_labels_path, "r") as file:
    strong_preds = json.load(file)

weak_ds = load_from_disk(args.weak_labels_path)
weak_labels = weak_ds['hard_label']
gt_labels = weak_ds['gt_label']

num_epochs = int(len(strong_preds) / len(weak_labels))
weak_labels = weak_labels * num_epochs
gt_labels = gt_labels * num_epochs

if args.batch_size:
    bs = int(args.batch_size)
else:
    bs = len(strong_preds)

nsteps = len(strong_preds) // bs
p_strong_eq_weak_wrongs = []
p_strong_eq_weak_rights = []
p_strong_eq_weaks = []
p_strong_eq_gts = []
steps = []

for i in range(nsteps + 1):
    start = bs * i
    end = bs * (i + 1)

    batch_strong = strong_preds[start:end]
    batch_weak = weak_labels[start:end]
    batch_gt = gt_labels[start:end]

    df_batch = pd.DataFrame(batch_strong, columns=['strong_pred'])
    df_batch['weak_label'] = batch_weak
    df_batch['gt_label'] = batch_gt

    df_batch_wrong = df_batch.loc[
        df_batch['strong_pred'] != df_batch['gt_label']
    ]
    df_batch_right = df_batch.loc[
        df_batch['strong_pred'] == df_batch['gt_label']
    ]

    # counting strong = weak | strong != gt
    n_wrong = len(df_batch_wrong)
    if n_wrong == 0:
        p_strong_eq_weak_wrong = 0
    else:
        n_strong_eq_weak_wrong = len(df_batch_wrong.loc[
            df_batch_wrong['strong_pred'] == df_batch_wrong['weak_label']
        ])
        p_strong_eq_weak_wrong = n_strong_eq_weak_wrong / n_wrong

    # counting strong = weak | strong = gt
    n_right = len(df_batch_right)
    if n_right == 0:
        p_strong_eq_weak_right = 0
    else:
        n_strong_eq_weak_right = len(df_batch_right.loc[
            df_batch_right['strong_pred'] == df_batch_right['weak_label']
        ])
        p_strong_eq_weak_right = n_strong_eq_weak_right / n_right

    # counting strong = gt and strong = weak
    n_strong_eq_weak = len(df_batch.loc[
        df_batch['strong_pred'] == df_batch['weak_label']
    ])
    n_strong_eq_gt = len(df_batch.loc[
        df_batch['strong_pred'] == df_batch['gt_label']
    ])
    p_strong_eq_weak = n_strong_eq_weak / bs
    p_strong_eq_gt = n_strong_eq_gt / bs

    p_strong_eq_weak_wrongs.append(p_strong_eq_weak_wrong)
    p_strong_eq_weak_rights.append(p_strong_eq_weak_right)
    p_strong_eq_weaks.append(p_strong_eq_weak)
    p_strong_eq_gts.append(p_strong_eq_gt)
    steps.append(i)

save_folder = os.path.dirname(args.strong_labels_path)
save_folder = os.path.basename(save_folder)
save_path = os.path.join(args.save_path, save_folder)
os.makedirs(save_path, exist_ok=True)

plt.plot(steps, p_strong_eq_weak_wrongs)
plt.plot(steps, p_strong_eq_weak_rights)
plt.xlabel('Training step', fontsize=12)
plt.ylabel('P(strong = weak | X)', fontsize=12)
plt.legend(['X: strong != gt', 'X: strong = gt'])
plt.savefig(os.path.join(save_path, "strong_weak_cond.jpg"), bbox_inches='tight')
plt.close()

plt.plot(steps, p_strong_eq_weaks)
plt.plot(steps, p_strong_eq_gts)
plt.xlabel('Training step', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.legend(['Strong = weak', 'Strong = GT'])
plt.savefig(os.path.join(save_path, "strong_weak_gt.jpg"), bbox_inches='tight')
plt.close()