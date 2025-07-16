import argparse
import json
import numpy as np
import os
from jinja2 import Environment, FileSystemLoader

# HTML 模板，可以保存为 template.html 文件
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Metrics Visualization for {{ data_name }} - Q: {{ question_idx }}</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        .container { border: 1px solid #ccc; padding: 1em; margin-bottom: 2em; border-radius: 8px; }
        .question { background-color: #f0f0f0; padding: 1em; border-radius: 5px; }
        .answer-segment { display: inline; }
        .tooltip { position: relative; display: inline-block; border-bottom: 1px dotted black; }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 250px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <h1>Visualization for {{ data_name }}: Question {{ question_idx }}</h1>
    
    <div class="container">
        <h2>Question</h2>
        <pre class="question">{{ question }}</pre>
    </div>

    <div class="container">
        <h2>Generated Answer (Colored by Metrics)</h2>
        <p>Hover over colored text to see metric values.</p>
        <pre>{{ colored_answer | safe }}</pre>
    </div>

</body>
</html>
"""

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def value_to_color(value, vmin, vmax, cmap='coolwarm'):
    """Maps a value to a color in HSL format."""
    if vmin == vmax:
        return 'hsl(0, 0%, 90%)' # Neutral color
    
    # Normalize value to 0-1 range
    norm_value = (value - vmin) / (vmax - vmin)
    
    # Coolwarm colormap (blue -> white -> red)
    # Hues: blue is around 240, red is 0
    hue = 240 - (norm_value * 240)
    saturation = 80
    lightness = 95 - (abs(norm_value - 0.5) * 80) # Darker at extremes, lighter in middle
    
    return f'hsl({hue}, {saturation}%, {lightness}%)'

def main():
    parser = argparse.ArgumentParser(description="Visualize representation metrics for a single sample.")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory where evaluation results are stored.")
    parser.add_argument("--run_name", type=str, required=True, help="The full name of the run, e.g., 'test_tool-integrated_-1_seed0_t0_s0_e-1'")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset (e.g., 'gsm8k').")
    parser.add_argument("--question_idx", type=str, required=True, help="The 'idx' of the question to visualize.")
    parser.add_argument("--layer", type=str, default="1", help="Which hidden layer's metrics to visualize.")
    args = parser.parse_args()

    # Construct file paths
    main_file = os.path.join(args.results_dir, args.data_name, f"{args.run_name}.jsonl")
    metrics_file = os.path.join(args.results_dir, args.data_name, f"{args.run_name}_repr_metrics.jsonl")
    
    if not os.path.exists(main_file) or not os.path.exists(metrics_file):
        print(f"Error: Result files not found!")
        print(f"Looked for: {main_file}")
        print(f"And: {metrics_file}")
        return

    # Load data
    main_data = {item['idx']: item for item in load_jsonl(main_file)}
    metrics_data = {item['idx']: item for item in load_jsonl(metrics_file)}
    
    # Find the specific sample
    if args.question_idx not in main_data or args.question_idx not in metrics_data:
        print(f"Error: Question with idx '{args.question_idx}' not found in result files.")
        return
        
    sample = main_data[args.question_idx]
    sample_metrics = metrics_data[args.question_idx]['metrics']
    
    # --- Prepare data for HTML template ---
    question_text = sample['question']
    # We visualize the first sampled answer
    answer_text = sample['code'][0]
    
    # Get metrics for the chosen layer
    layer_metrics = sample_metrics.get(args.layer)
    if not layer_metrics:
        print(f"Error: Metrics for layer '{args.layer}' not found.")
        return

    # Find min/max for color normalization across all metrics for this sample
    all_values = []
    for name, values in layer_metrics.items():
        all_values.extend(values)
    vmin, vmax = min(all_values), max(all_values)

    # Split answer into segments and color them
    tokens = answer_text.split() # A simple whitespace tokenizer
    stride = len(tokens) // len(layer_metrics.get('Response Entropy 1 diff', [1])) # Infer stride
    
    colored_html = ""
    for i in range(len(layer_metrics.get('Response Entropy 1 diff', []))):
        start_token = i * stride
        end_token = (i + 1) * stride
        segment_text = " ".join(tokens[start_token:end_token])
        
        tooltip_lines = []
        avg_metric_val = 0
        metric_count = 0
        for metric_name in sorted(layer_metrics.keys()):
            if i < len(layer_metrics[metric_name]):
                val = layer_metrics[metric_name][i]
                tooltip_lines.append(f"<b>{metric_name}</b>: {val:.4f}")
                avg_metric_val += val
                metric_count += 1
        
        avg_metric_val = avg_metric_val / metric_count if metric_count > 0 else 0
        bg_color = value_to_color(avg_metric_val, vmin, vmax)
        
        tooltip_text = "<br>".join(tooltip_lines)
        
        colored_html += (
            f'<span class="tooltip" style="background-color: {bg_color};">'
            f'{segment_text} '
            f'<span class="tooltiptext">{tooltip_text}</span>'
            f'</span>'
        )

    # Render HTML
    env = Environment(loader=FileSystemLoader('.'))
    # Create a template file from the string
    with open("template.html", "w") as f:
        f.write(HTML_TEMPLATE)
    template = env.get_template("template.html")
    
    output_html = template.render(
        data_name=args.data_name,
        question_idx=args.question_idx,
        question=question_text,
        colored_answer=colored_html
    )

    # Save and report
    output_file = f"demo_{args.data_name}_{args.question_idx}.html"
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(output_html)
        
    print(f"\nVisualization saved to '{output_file}'. Open this file in your browser.")


if __name__ == "__main__":
    main()