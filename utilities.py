import numpy as np


def preprocess_data(methyl_data, labels, label_categories = None):
    #take out unlabeled/extra labeled samples
    features = []
    new_labels = []
    categories = set()
    
    if not label_categories == None:
        for i in range(len(labels)):
            if labels[i] != 'other' and label[i] in S:
                new_labels.append(labels[i])
                features.append(methyl_data[i])
                categories.add(labels[i])
    else:
        for i in range(len(labels)):
            if labels[i] != 'other':
                new_labels.append(labels[i])
                features.append(methyl_data[i])
                categories.add(labels[i])
    features = np.array(features)
    new_labels = np.array(new_labels)

    print("All Labels:", ", ".join(str(e) for e in categories))
    
    colors = ['#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2',
           '#b5bd61', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
           '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d']
    label_to_color = {}
    for color_index, l in enumerate(sorted(categories)):
        label_to_color[l] = colors[color_index]
    label_colors = np.copy(new_labels)
    for i in range(len(new_labels)):
        label_colors[i] = label_to_color[new_labels[i]]
    
    return (features, new_labels, label_colors, label_to_color)
