import os
import json
import glob
from collections import defaultdict
from pathlib import Path
from datetime import datetime


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib import ticker 

# Read data from data.json
with open("template/data.json", "r") as f:
    data = json.load(f)

model_colors = {# ff8700, fbcd2b, 00ffdb, 00ff5c
    "gpt-4o-mini-2024-07-18": "#ffdc00", # TODO: we don't have input/output tokens right now
    "gpt-4o-2024-05-13": "#ffdc00", # TODO: we don't have input/output tokens right now
    "gemini-1.5-flash-002": "#4ecc30", 
    "gemini-1.5-pro-002": "#4ecc30", 
    "gemini-2.5-pro-exp-03-25": "#4ecc30", 
    "llama-3.2-1b-it": "#1e93ff",
    "llama-3.2-3b-it": "#1e93ff",
    "llama-3.1-8b-it": "#1e93ff",
    "llama-3.1-70b-it": "#1e93ff", # TODO: we don't have input/output tokens right now
    # "llama-3.2-11b-it": "#ff00fd",
    # "llama-3.2-90b-it": "#ff007e",
    "claude-3.5-haiku-2024-10-22": "#f93c32",
    "claude-3.5-sonnet-2024-10-22": "#f93c32",
    # "o1-preview": "#000000",
    "reka-flash-3": "#ff00fd",
    "deepseek-r1": "#ff841c",
    "grok-3-beta": "#008080",
}

model_name_dictionary = {
    "claude-3.5-sonnet-2024-10-22": "claude-3.5-sonnet",
    "claude-3.5-haiku-2024-10-22": "claude-3.5-haiku",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4o-2024-05-13": "gpt-4o",
    "gemini-1.5-flash-002": "gemini-1.5-flash",
    "gemini-1.5-pro-002": "gemini-1.5-pro",
    "gemini-2.5-pro-exp-03-25": "gemini-2.5-pro",
    "llama-3.2-1b-it": "llama-3.2-1b-it",
    "llama-3.2-3b-it": "llama-3.2-3b-it",
    "llama-3.1-8b-it": "llama-3.1-8b-it",
    "llama-3.1-70b-it": "llama-3.1-70b-it",
    "llama-3.3-70b-it": "llama-3.3-70b-it",
    "llama-3.2-11b-it": "llama-3.2-11b-it",
    "llama-3.2-90b-it": "llama-3.2-90b-it",
    "o1-preview": "o1-preview",
    "reka-flash-3": "reka-flash-3",
    "deepseek-r1": "deepseek-r1",
    "grok-3-beta": "grok-3-beta",
}

# Process data and extract release dates
cost_per_1M_input_tokens = {
    "claude-3.5-sonnet-2024-10-22": 3.0,
    "claude-3.5-haiku-2024-10-22": 0.8,
    "gpt-4o-mini-2024-07-18": 0.15,
    "gpt-4o-2024-05-13": 2.5, 
    "gemini-1.5-flash-002": 0.075,
    "gemini-1.5-pro-002": 1.25,
    "gemini-2.5-pro-exp-03-25": 1.25,
    "llama-3.2-1b-it": 0.01,
    "llama-3.2-3b-it": 0.015,
    "llama-3.1-8b-it": 0.02,
    "llama-3.1-70b-it": 0.13,
    "llama-3.3-70b-it": 0.13,
    "llama-3.2-11b-it": 0.06,
    "llama-3.2-90b-it": 0.35,
    "o1-preview": 15.0,
    "reka-flash-3": 0.20,
    "deepseek-r1": 0.55,
    "grok-3-beta": 3.0
}



results = {"llm": {}, "vlm": {}}

include_models = list(model_colors.keys())
game_names_set = set()

for leaderboard in data["leaderboards"]:
    modality = leaderboard["name"].lower()  # 'llm' or 'vlm'
    for model_result in leaderboard["results"]:
        model_name = model_result["name"].lower()
        if model_name not in include_models:
            continue  # Skip models not in the include_models list
        for game in model_result:
            if game in [
                "name",
                "average",
                "folder",
                "date",
                "trajs",
                "site",
                "verified",
                "oss",
                "org_logo",
            ]:
                continue
            value, error, count = model_result[game]
            game_names_set.add(game)
            if game not in results[modality]:
                results[modality][game] = {}
            results[modality][game][model_name] = (value, error)


model_stats = {"llm": defaultdict(dict)}
for leaderboard in data["leaderboards"]:
    modality = leaderboard["name"].lower()  # 'llm' or 'vlm'
    # skip vlm
    if modality == "vlm":
        continue

    for model_result in leaderboard["results"]:
        model_name = model_result["name"].lower()
        if model_name not in include_models:
            continue  # Skip models not in the include_models list

        with open(Path(model_result["folder"]) / "summary.json", "r+") as f:
            model_summary = json.load(f)
            model_stats[modality][model_name]["total_input_tokens"] = model_summary["total_input_tokens"] + model_summary["total_output_tokens"] * 2
            model_stats[modality][model_name]["total_output_tokens"] = model_summary["total_output_tokens"]


def plot_average_progression(results, title, filename, model_colors):
    models = list(model_colors.keys())
    
    # Initialize data for averages
    avg_progression = {"llm": {}, "vlm": {}}

    # Compute averages for LLM and VLM
    for modality in ["llm", "vlm"]:
        for model in include_models:
            values = []
            for task, task_data in results[modality].items():
                if model in task_data:
                    values.append(task_data[model][0])
            if values:
                avg_progression[modality][model] = np.mean(values)
            else:
                avg_progression[modality][model] = None

    # Get dates and sort models chronologically
    total_cost = [model_stats["llm"][model]["total_input_tokens"] * cost_per_1M_input_tokens[model] * 1e-6 / 255 for model in models]
    sorted_indices = np.argsort(total_cost)
    sorted_models = [models[i] for i in sorted_indices]
    
    # Prepare data
    llm_values = [avg_progression["llm"].get(model, None) for model in sorted_models]
    vlm_values = [avg_progression["vlm"].get(model, None) for model in sorted_models]
    colors = [model_colors[model] for model in sorted_models]
    costs = np.array(total_cost)[sorted_indices]

    fig, ax = plt.subplots(figsize=(14, 8))
    # Set the plot background to transparent
    fig.patch.set_alpha(0)  # Makes the figure background transparent
    ax.set_facecolor((0, 0, 0, 0))  # Sets the axes background to transparent

    # Scatter plots
    ax.scatter(costs, llm_values, c=colors, edgecolor='black', s=250)
    # ax.scatter(costs, llm_values, c=colors, edgecolor='black', s=100, label='LLM')
    # ax.scatter(costs, vlm_values, c=colors, edgecolor='black', s=100, 
    #            marker='^', label='VLM')
    
    def format_with_dollar(x, pos):
        return f"${x}"

    color = "white"

    ax.set_xscale('log')
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.01, 0.1, 1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_with_dollar))
    ax.tick_params(colors=color, labelsize=18)
    ax.grid(color=color, linestyle='-', linewidth=0.5)
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5)
    ax.spines['bottom'].set_color(color)
    ax.spines['left'].set_color(color)
    ax.spines['top'].set_color((0, 0, 0, 0))
    ax.spines['right'].set_color((0, 0, 0, 0))

    for i, model in enumerate(sorted_models):
        if llm_values[i] is not None:
            # ax.text(costs[i], llm_values[i], model_name_dictionary[model], fontsize=8, ha="center")
            if model in ["gpt-4o-2024-05-13", "claude-3.5-haiku-2024-10-22", "grok-3-beta"]:
                ax.text(costs[i], llm_values[i] - 1.0, model_name_dictionary[model], fontsize=14, ha="center", va='top', color=color)
            elif model in [
                "llama-3.2-1b-it",
                "llama-3.2-3b-it",
                "llama-3.1-8b-it",
                
            ]:
                ax.text(costs[i] * 1.1, llm_values[i] + 1.0, model_name_dictionary[model], fontsize=14, ha="center", va='bottom', color=color)
            elif model in ["llama-3.1-70b-it"]:
                ax.text(costs[i], llm_values[i] - 1.0, model_name_dictionary[model], fontsize=14, ha="center", va='top', color=color)
            else:
                ax.text(costs[i], llm_values[i] + 1.0, model_name_dictionary[model], fontsize=14, ha="center", va='bottom', color=color)
        # if vlm_values[i] is not None:
        #     ax.text(costs[i], vlm_values[i], model_name_dictionary[model], fontsize=8, ha='right')

    # Legend and labels
    ax.legend()
    # ax.set_title("BALROG LEADERBOARD", color=color)
    ax.set_ylabel("SCORE (%)", color=color, fontsize=18)
    ax.set_xlabel("COST PER EPISODE ($)", color=color, fontsize=18)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot and save the average progression results
plot_average_progression(
    results,
    "Average Progression for LLM and VLM",
    "average_progression.png",
    model_colors,
)
