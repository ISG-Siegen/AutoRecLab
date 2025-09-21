from treesearch.backend.utils import FunctionSpec

vlm_feedback_spec = FunctionSpec(
    name="analyze_experiment_plots",
    json_schema={
        "type": "object",
        "properties": {
            "plot_analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "string",
                            "description": "Detailed analysis of the plot's results and implications",
                        },
                    },
                    "required": ["analysis"],
                },
            },
            "valid_plots_received": {
                "type": "boolean",
                "description": "True if valid plots were received, False otherwise. For example, if the plots are empty or not meaningful, this should be False.",
            },
            "vlm_feedback_summary": {
                "type": "string",
                "description": "Summarize the feedback from the VLM. If the task involves generative modeling, make sure to focus on the generated samples.",
            },
        },
        "required": ["plot_analyses", "valid_plots_received", "vlm_feedback_summary"],
    },
    description="Analyze experimental plots and provide detailed feedback on the results.",
)

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, summarize the bug and propose a fix. Otherwise, leave it empty.",
            },
        },
        "required": [
            "is_bug",
            "summary",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

plot_selection_spec = FunctionSpec(
    name="select_plots",
    json_schema={
        "type": "object",
        "properties": {
            "selected_plots": {
                "type": "array",
                "description": "List of selected plot file paths",
                "items": {"type": "string", "description": "Full path to a plot file"},
                "maxItems": 10,
            }
        },
        "required": ["selected_plots"],
    },
    description="Select the 10 most relevant plots for analysis",
)
