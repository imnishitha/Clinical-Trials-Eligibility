import plotly.graph_objects as go
import pandas as pd
import os
from collections import Counter

def create_label_distribution_plots(train_labels, val_labels, test_labels, save_path="./Plots/training_class_distribution.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    all_labels = ["Negative", "Neutral", "Positive"]
    label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def compute_percentage_distribution(labels):
        mapped = [label_mapping[i] for i in labels]
        counter = Counter(mapped)
        total = sum(counter.values())
        return {label: round((counter[label] / total) * 100, 2) for label in all_labels}
    
    train_dist = compute_percentage_distribution(train_labels)
    val_dist = compute_percentage_distribution(val_labels)
    test_dist = compute_percentage_distribution(test_labels)

    df = pd.DataFrame({
        "Label": all_labels,
        "Train": [train_dist[label] for label in all_labels],
        "Validation": [val_dist[label] for label in all_labels],
        "Test": [test_dist[label] for label in all_labels],
    })

    fig = go.Figure(data=[
        go.Bar(name="Negative", x=["Train", "Validation", "Test"], y=[df.loc[0, "Train"], df.loc[0, "Validation"], df.loc[0, "Test"]], 
               text=[f"{df.loc[0, 'Train']}%", f"{df.loc[0, 'Validation']}%", f"{df.loc[0, 'Test']}%"], textposition="inside"),
        go.Bar(name="Neutral",  x=["Train", "Validation", "Test"], y=[df.loc[1, "Train"], df.loc[1, "Validation"], df.loc[1, "Test"]],
               text=[f"{df.loc[1, 'Train']}%", f"{df.loc[1, 'Validation']}%", f"{df.loc[1, 'Test']}%"], textposition="inside"),
        go.Bar(name="Positive", x=["Train", "Validation", "Test"], y=[df.loc[2, "Train"], df.loc[2, "Validation"], df.loc[2, "Test"]],
               text=[f"{df.loc[2, 'Train']}%", f"{df.loc[2, 'Validation']}%", f"{df.loc[2, 'Test']}%"], textposition="inside"),
    ])

    fig.update_layout(
        barmode='stack',
        title="Class Distribution (%) Across Dataset Splits",
        xaxis_title="Dataset Split",
        yaxis_title="Percentage of Samples",
        yaxis=dict(range=[0, 100]),
    )

    fig.write_image(save_path)
    print(f"Class Distribution plot saved to {save_path}")