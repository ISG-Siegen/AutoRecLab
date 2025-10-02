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

score_code_func_spec = FunctionSpec(
    name="score_code",
    json_schema={
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "A percentage score grading the quality of the code implementation. A higher percentage indicates better quality.",
                "minimum": 0,
                "maximum": 100,
            },
            "is_satisfactory": {
                "type": "boolean",
                "description": "true if the code implementation is satisfactory and meets all requirements, otherwise false.",
            },
            "feedback": {
                "type": "string",
                "description": "A short general feedback and the list of all requirements and whether they are fulfilled or not.",
            },
        },
        "required": ["score", "is_satisfactory", "feedback"],
    },
    description="Score the code implementation and provide feedback on its quality.",
)

# TODO: Change requirements from string to dict with requirement and boolean or int (0/1)
set_code_requirements_spec = FunctionSpec(
    name="set_code_requirements",
    json_schema={
        "type": "object",
        "properties": {
            "requirements": {
                "type": "string",
                "description": "A numbered list of clear, specific code requirements.",
            }
        },
        "required": ["requirements"],
    },
    description=(
        "Set clear and specific code requirements for the implementation based on the research task."
    ),
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
