#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics
from statannotations.Annotator import Annotator

sns.set_style('whitegrid')


# In[2]:


df = pd.read_csv('/srv/store/Projects/schan/quantum/mimic/mimic_test.csv')
df = df[df['ViewPosition'].isin(['AP', 'PA'])]
# df.to_csv('/srv/store/Projects/schan/quantum/mimic/test_clean.csv', index=False)


# In[3]:


# All the labels
global_labels = {
    8: np.sort(
        [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
        ]
    ).tolist(),
    14: np.sort(
        [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural Thickening',
            'Hernia',
        ]
    ).tolist(),
    19: np.sort(
        [
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Effusion',
            'Emphysema',
            'Fibrosis',
            'Hernia',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pleural Thickening',
            'Pneumonia',
            'Pneumothorax',
            'Pneumoperitoneum',
            'Pneumomediastinum',
            'Subcutaneous Emphysema',
            'Tortuous Aorta',
            'Calcification of the Aorta',
        ]
    ).tolist(),
}


# In[4]:


def youden_threshold(y_true, y_pred):
    # Youden's J Statistic threshold
    fprs, tprs, thresholds = metrics.roc_curve(y_true, y_pred)
    return thresholds[np.nanargmax(tprs - fprs)]


# In[11]:


# Load ground-truth
df_nih_test = pd.read_csv(
    '/srv/store/Projects/schan/quantum/PruneCXRlabels/miccai2023_nih-cxr-lt_labels_test.csv'
)
df_mimic_test = pd.read_csv('/srv/store/Projects/schan/quantum/mimic/mimic_test.csv')
# Harmonize the mimic csv
df_mimic_test['id'] = df_mimic_test['dicom_id'] + '.jpg'
t = list(df_mimic_test.columns)
t[t.index('Pleural Effusion')] = 'Effusion'
df_mimic_test.columns = t
# Order evertyhing
df_nih_test = df_nih_test[['id'] + global_labels[19]].sort_values('id')
df_mimic_test = df_mimic_test[['id'] + global_labels[19]].sort_values('id')


# In[12]:


overall_res = []
detailed_res = []
for num_labels in [8, 14, 19]:
    labels = global_labels[num_labels]
    for model in ['classical', 'quantum']:
        for seed in [1, 2, 3, 4, 5]:
            # Load predictions
            if model == 'classical':
                num_layers = 0
            else:
                num_layers = 3

            #### NIH
            test_ds = 'nih'
            y_pred = pd.read_csv(
                f'experiments/ISVLSI/classifier={model}-fraction=1.0-freeze=False-num_labels={num_labels}/learning_rate=0.0001-num_layers={num_layers}-seed={seed}/results/preds.csv'
            )
            # Harmonize nih preds
            t = list(y_pred.columns)
            t[0] = 'path'
            y_pred.columns = t

            # Keep id and labels only. Remove everything else
            # This also orders everything
            y_pred['id'] = y_pred['path'].str.split('/').str[-1]
            y_pred = y_pred.sort_values('id')
            y_pred = y_pred[labels]

            # Load ground-truth
            y_true = df_nih_test[labels]

            # Sanity check!
            if y_pred.isnull().any().any():
                print(model, num_labels, seed)
                continue

            # # Calculuate threshold on NIH test only
            # # Use these for MIMIC
            thresholds = []
            # Calculate metrics
            aucs = []
            accs = []
            # fprs = []
            # fnrs = []
            for label in labels:
                y_true_t = y_true[label].values
                y_pred_t = y_pred[label].values

                threshold = youden_threshold(y_true_t, y_pred_t)
                y_clf_t = (y_pred_t > threshold).astype(int)

                auc_t = metrics.roc_auc_score(y_true_t, y_pred_t)
                acc_t = metrics.balanced_accuracy_score(y_true_t, y_clf_t)
                # fpr_t = 1 - metrics.recall_score(y_true_t, y_clf_t, pos_label=0)
                # fnr_t = 1 - metrics.recall_score(y_true_t, y_clf_t)

                aucs += [auc_t]
                accs += [acc_t]
                # fprs += [fpr_t]
                # fnrs += [fnr_t]
                thresholds += [threshold]
                detailed_res += [[model, num_labels, test_ds, seed, label, auc_t, acc_t]]

            overall_res += [
                [
                    model,
                    num_labels,
                    test_ds,
                    seed,
                    np.mean(aucs),
                    np.std(aucs),
                    np.mean(accs),
                    np.std(accs),
                ]
            ]

            #### MIMIC
            test_ds = 'mimic'
            y_pred = pd.read_csv(
                f'experiments/ISVLSI/mimic/classifier={model}-fraction=1.0-freeze=False-num_labels={num_labels}/learning_rate=0.0001-num_layers={num_layers}-seed={seed}/mimic_preds.csv'
            )
            # Harmonize mimic preds
            t = list(y_pred.columns)
            t[t.index('Pleural Effusion')] = 'Effusion'
            y_pred.columns = t

            # Keep id and labels only. Remove everything else
            # This also orders everything
            y_pred['id'] = y_pred['path'].str.split('/').str[-1]
            y_pred = y_pred.sort_values('id')
            y_pred = y_pred[labels]

            # Load ground-truth
            y_true = df_mimic_test[labels]

            # Calculate metrics
            aucs = []
            accs = []
            # fprs = []
            # fnrs = []
            for i, label in enumerate(labels):
                y_true_t = y_true[label].values
                y_pred_t = y_pred[label].values

                y_clf_t = (y_pred_t > thresholds[i]).astype(int)
                auc_t = metrics.roc_auc_score(y_true_t, y_pred_t)
                acc_t = metrics.balanced_accuracy_score(y_true_t, y_clf_t)
                # fpr_t = 1 - metrics.recall_score(y_true_t, y_clf_t, pos_label=0)
                # fnr_t = 1 - metrics.recall_score(y_true_t, y_clf_t)

                aucs += [auc_t]
                accs += [acc_t]
                # fprs += [fpr_t]
                # fnrs += [fnr_t]
                detailed_res += [[model, num_labels, test_ds, seed, label, auc_t, acc_t]]

            overall_res += [
                [
                    model,
                    num_labels,
                    test_ds,
                    seed,
                    np.mean(aucs),
                    np.std(aucs),
                    np.mean(accs),
                    np.std(accs),
                ]
            ]

pd.DataFrame(
    np.array(detailed_res),
    columns=['model', 'num_labels', 'test_ds', 'seed', 'label', 'auc', 'acc'],
).to_csv('main_results_detailed.csv', index=False)
pd.DataFrame(
    np.array(overall_res),
    columns=['model', 'num_labels', 'test_ds', 'seed', 'auc_mean', 'auc_sd', 'acc_mean', 'acc_sd'],
).to_csv('main_results.csv', index=False)


# In[11]:


df = pd.read_csv('main_results.csv')

metric = 'auc'
for test_ds in ['nih', 'mimic']:
    df_t = df[df.test_ds == test_ds]

    plt.figure(figsize=(5, 4))
    ax = sns.barplot(data=df_t, x='num_labels', y=f'{metric}_mean', hue='model', ci='sd')
    # Fix labels
    ax.legend_.set_title('Model')
    new_labels = ['CDL', 'DQC']
    for t, l in zip(ax.legend_.texts, new_labels):
        t.set_text(l)

    pairs = [
        ((8, 'classical'), (8, 'quantum')),
        ((14, 'classical'), (14, 'quantum')),
        ((19, 'classical'), (19, 'quantum')),
    ]

    for (a, b), (x, y) in pairs:
        c = df_t[(df_t.num_labels == a) & (df_t.model == b)][[f'{metric}_mean']].values
        z = df_t[(df_t.num_labels == x) & (df_t.model == y)][[f'{metric}_mean']].values
        print(test_ds, a, b, np.round(c.mean(), 2), np.round(c.std(), 2))
        print(test_ds, x, y, np.round(z.mean(), 2), np.round(z.std(), 2))

    annot = Annotator(
        ax, pairs, data=df_t, x='num_labels', y=f'{metric}_mean', hue='model', ci='sd'
    )
    annot.configure(
        test='t-test_paired',
        text_format='star',
        verbose=1,
        pvalue_thresholds=[[1e-3, '***'], [1e-2, '**'], [0.05, '*'], [1, 'ns']],
    )
    annot.apply_test().annotate()

    plt.xlabel('Classification Task')
    if metric == 'auc':
        plt.ylabel('AUROC')
    else:
        plt.ylabel('Balanced Accuracy')
    plt.title(f'{test_ds.upper()}-CXR-LT Test Set')
    plt.xticks([0, 1, 2], ['CXR-8', 'CXR-14', 'CXR-19'])
    # plt.savefig(f'figures/{test_ds}_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# In[32]:


df = pd.read_csv('main_results.csv')

for num_labels in [8, 14, 19]:
    df_t = df[df.num_labels == num_labels]
    subtract = lambda x: x.iloc[0] - (x.iloc[1] if len(x) == 2 else 0)
    df_t = (
        df_t[['seed', 'test_ds', 'auc_mean']]
        .groupby(['test_ds', 'seed'])
        .apply(subtract)
        .reset_index()
    )
    x = df_t[df_t.test_ds == 'nih'].auc_mean.values
    y = df_t[df_t.test_ds == 'mimic'].auc_mean.values
    print(num_labels)
    print(x.mean(), x.std())
    print(y.mean(), y.std())
    print(stats.shapiro(x - y), stats.ttest_rel(x, y))


# In[27]:


df_t


# In[5]:


label_order = pd.read_csv('/srv/store/Projects/schan/quantum/cxr_lt_label_counts.csv')
label_order['NIH'] = label_order[['NIH Train', 'NIH Val', 'NIH Test']].sum(axis=1)
label_order = label_order.sort_values('NIH', ascending=False)
# label_order = label_order['Label'].tolist()
label_order


# In[20]:


df = pd.read_csv('main_results_detailed.csv')

test_ds = 'nih'
metric = 'auc'
for test_ds in ['nih', 'mimic']:
    for num_labels in [8, 14, 19]:
        labels = global_labels[num_labels]
        order = label_order[label_order['Label'].isin(labels)]['Label'].tolist()

        df_t = df[(df.test_ds == test_ds) & (df.num_labels == num_labels)]

        # plt.figure(figsize=(4,5))
        plt.figure(figsize=(5, 7))
        ax = sns.barplot(data=df_t, y='label', x=metric, hue='model', ci='sd', order=order)
        # Fix labels
        ax.legend(loc='upper left')
        ax.legend_.set_title('Model')
        new_labels = ['CDL', 'DQC']
        for t, l in zip(ax.legend_.texts, new_labels):
            t.set_text(l)

        pairs = [((label, 'classical'), (label, 'quantum')) for label in labels]

        annot = Annotator(
            ax, pairs, data=df_t, y='label', x=metric, hue='model', ci='sd', orient='h'
        )
        annot.configure(
            test='t-test_paired',
            text_format='star',
            verbose=2,
            pvalue_thresholds=[[1e-3, '***'], [1e-2, '**'], [0.05, '*'], [1, 'ns']],
        )
        annot.apply_test().annotate()

        plt.ylabel('Label')
        if metric == 'auc':
            plt.xlabel('AUROC')
        else:
            plt.xlabel('Balanced Accuracy')
        plt.title(f'{test_ds.upper()}-CXR-LT Test Set')
        plt.savefig(f'figures/{test_ds}_{metric}_{num_labels}.jpg', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()


# In[5]:


df = pd.DataFrame(columns=['model', 'num_labels', 'seed', 'epoch', 'train_loss', 'val_loss'])
for num_labels in [8, 14, 19]:
    for model in ['classical', 'quantum']:
        for seed in [3]:
            # Load predictions
            if model == 'classical':
                num_layers = 0
            else:
                num_layers = 3

            try:
                history = pd.read_csv(
                    f'experiments/ISVLSI/classifier={model}-fraction=1.0-freeze=False-num_labels={num_labels}/learning_rate=0.0001-num_layers={num_layers}-seed={seed}/results/history.csv'
                )[['epoch', 'train_loss', 'val_loss']]
            except:
                print(num_labels, model, seed)
                continue
            history['model'] = model
            history['num_labels'] = num_labels
            history['seed'] = seed
            df = pd.concat((df, history))
df = df.reset_index(drop=True)

df_train = df[['model', 'num_labels', 'seed', 'epoch', 'train_loss']]
df_train['type'] = 'train'
df_train['loss'] = df_train['train_loss']
df_train = df_train.drop('train_loss', axis=1)
df_val = df[['model', 'num_labels', 'seed', 'epoch', 'val_loss']]
df_val['type'] = 'val'
df_val['loss'] = df_val['val_loss']
df_val = df_val.drop('val_loss', axis=1)
df = pd.concat((df_train, df_val)).reset_index(drop=True)

for num_labels in [8]:  # , 14, 19]:
    df_t = df[df.num_labels == num_labels]
    plt.figure(figsize=(5, 4))
    ax = sns.lineplot(data=df_t, x='epoch', y='loss', hue='model', style='type')
    plt.legend(bbox_to_anchor=(1.5, 1))
    new_labels = ['Model', 'CDL', 'DQC', 'Loss', 'Train', 'Val']
    for i, label in enumerate(ax.get_legend().get_texts()):
        label.set_text(new_labels[i])
    plt.title(f'No. of Labels = {num_labels}')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.savefig(f'train_loss_{num_labels}_old.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# In[15]:


df_train = pd.DataFrame(columns=['model', 'num_labels', 'seed', 'step', 'train_loss'])
seed = 42
for num_labels in [8, 14, 19]:
    for model in ['classical', 'quantum']:
        # Load predictions
        if model == 'classical':
            num_layers = 0
        else:
            num_layers = 3

        try:
            history = pd.read_csv(
                f'monitorloss/classifier={model}-fraction=1.0-freeze=False-num_labels={num_labels}/learning_rate=0.0001-num_layers={num_layers}-seed={seed}/results/train_losses_over_steps.txt',
                header=None,
            )
            history.columns = ['train_loss']
            history['step'] = np.arange(1, len(history) + 1)
            history.iloc[1:, 1] = (history.iloc[1:, 1] // 256 + 1) * 256
        except:
            print(num_labels, model, seed)
            continue
        history['model'] = model
        history['num_labels'] = num_labels
        history['seed'] = seed
        df_train = pd.concat((df_train, history))
df_train = df_train.reset_index(drop=True)

df_val = pd.DataFrame(columns=['model', 'num_labels', 'seed', 'epoch', 'val_loss'])
for num_labels in [8, 14, 19]:
    for model in ['classical', 'quantum']:
        # Load predictions
        if model == 'classical':
            num_layers = 0
        else:
            num_layers = 3

        try:
            history = pd.read_csv(
                f'monitorloss/classifier={model}-fraction=1.0-freeze=False-num_labels={num_labels}/learning_rate=0.0001-num_layers={num_layers}-seed={seed}/results/history.csv'
            )[['epoch', 'val_loss']]
        except:
            print(num_labels, model, seed)
            continue
        history['step'] = (history['epoch'] + 1) * 2454
        history['model'] = model
        history['num_labels'] = num_labels
        history['seed'] = seed
        df_val = pd.concat((df_val, history))
df_val = df_val.reset_index(drop=True)

for num_labels in [8, 14, 19]:
    df_train_t = df_train[df_train.num_labels == num_labels]
    df_val_t = df_val[df_val.num_labels == num_labels]
    plt.figure(figsize=(6, 4))
    ax = sns.lineplot(data=df_train_t, x='step', y='train_loss', hue='model', ci='sd')
    ax = sns.lineplot(
        ax=ax,
        data=df_val_t,
        x='step',
        y='val_loss',
        hue='model',
        ci='sd',
        linestyle='--',
        legend=None,
    )
    ax.legend_.set_title('Model')
    new_labels = ['CDL', 'DQC']
    for t, l in zip(ax.legend_.texts, new_labels):
        t.set_text(l)
    plt.title(f'CXR-{num_labels}')
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.xlim((-5000, 127500))
    plt.ylim((0.05, 0.75))
    plt.savefig(f'figures/train_loss_{num_labels}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# In[79]:


df_train


# In[80]:


df_val
