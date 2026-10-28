# AI Usage Disclosure

<a id="sec:ai_usage"></a>

The group takes full responsibility for all submitted code, numerical results, figures, claims, and text. AI tools supported learning, reviewing and refining member-written code and drafts, building skeleton pages, and language editing. Group members reviewed the adopted suggestions and verified them against the implementation, experimental outputs, and assignment requirements. All reported experimental results were obtained from runs executed by the group, and result figures were produced from those outputs. AI assistance with code or presentation did not replace the execution of experiments or provide fabricated data or results. The following records describe the scope of AI assistance and the corresponding student verification.

## Review and Refinement of the Training and Evaluation Pipeline

- **Tool:** GPT-5.6 Sol.

- **Used by:** Lê Nguyễn Gia Phúc.

- **Development stage:** Pipeline review, refinement, and debugging (Milestone 1).

- **Task:** Reviewing and refining a member-written draft of the FashionMNIST pipeline, covering data preparation, training, evaluation, checkpoint selection, and reproducibility.

- **Prompt summary:** Provided an existing member-written pipelineand specified the required workflow, current implementation of EDA,train/validation/test splitting, preprocessingand augmentation, model training pipeline,evaluation metrics, checkpoint selection,and reproducibility.Requested a review of the workflow,explanations of technical issues,and suggestions for improving reproducibility.Also requested appropriate ways to organizeand save checkpoints, configurations,experimental outputs.

- **AI contribution:** Reviewed the supplied pipeline and suggested improvements to data preparation, training, evaluation, checkpoint selection, and reproducibility. Recommended saving model configurations, dataset split indices, seed settings, dependency versions, hardware information, checkpoints, metric summaries, and training histories. Suggested linking these artifacts through experiment IDs to support result traceability. Helped modify visualization code for learning curves, confusion matrices, and prediction examples.

- **Student verification:** Compared the suggested changes with the original member-written pipeline and the experimental requirements. Reviewed the data split, preprocessing, training settings, metric calculations, checkpoint-selection rule, and reproducibility settings. Executed the training workflow and inspected the generated metrics and checkpoints.

- **Affected files/sections:** Assignment1/Assignment1.ipynb.

- **Responsible member:** Lê Nguyễn Gia Phúc.

## Student-Proposed Architectures and Technical Verification

- **Tool:** GPT-5.6 Sol

- **Development stage**: Model Implementation (Milestone 1)

- **Used by:** Lê Nguyễn Gia Phúc

- **Task:** Reviewing the student-proposed Linear, MLP, CNN, LSTM, and Transformer architectures for implementation consistency and correctness.

- **Prompt summary:** Proposed the five model architectures and their associated input representations: flattened pixels for Linear and MLP, two-dimensional images for CNN, row sequences for LSTM, and patch tokens for Transformer. Provided model code and requested checks of tensor dimensions, layer connections, dropout placement, and consistency between the model parameters.

- **AI contribution:** Reviewing the configurations, explaining tensor dimensions and layer connections, and suggesting dropout probabilities and placement for regularization. Checked parameter-count calculations and confirmed that the four nonlinear models have approximately comparable trainable parameter counts

- **Student verification:** Compared the programmatically counted trainable parameters across the four nonlinear models to verify that their parameter counts were approximately comparable. Checked that the implemented dropout probabilities and locations matched the selected configurations, and evaluated their combined effect through the dropout ablation study.

- **Affected files/sections:** Assignment1/Assignment1.ipynb.

- **Responsible member:** Lê Nguyễn Gia Phúc.

## Feedback on Results Interpretation and Discussion

- **Tool:** GPT-5.6 Sol

- **Development stage**: Report writing (Milestone 2)

- **Used by:** Lê Nguyễn Gia Phúc.

- **Task:** Reviewing the student's initial interpretations and suggesting improvements to the organization and focus of the comparative discussion.

- **Prompt summary:** Provided the assignment's discussion requirements, the experimental results already obtained, and an initial student-written draft explaining those results. Requested feedback on the draft and suggestions on how to organize and refine the explanations in the report, linking relevant comparisons, observed patterns, and supporting evidence to each requirement. The aim was to keep the discussion focused on the specification and available evidence, while avoiding repetitive descriptions, irrelevant commentary, and unsupported conclusions.

- **AI contribution:** Reviewed the supplied draft and suggested revisions to its structure, wording, and interpretation of the results. Helped connect overall performance, class-specific differences, confusion patterns, training behavior, and computational trade-offs to the assignment requirements. Suggested qualifying explanations related to architecture, regularization, and training where the experiments did not isolate their individual effects.

- **Student verification:** Compared the suggested revisions with the original draft and the assignment's discussion requirements. Checked numerical comparisons against notebook outputs and result tables, and assessed interpretations against the learning curves, confusion matrices, and prediction examples.

- **Affected files/sections:** Report(Implementation Results)

- **Responsible member:** Lê Nguyễn Gia Phúc.

## English Language Editing

- **Tool:** GPT-5.6 Sol.

- **Development stage**: Final Revision (Milestone 2)

- **Used by:** Hoàng Xuân Bách,Lê Nguyễn Gia Phúc, Nguyễn Việt Hùng

- **Task:** Improving the grammar, clarity, and readability of student-written explanations in the report.

- **Prompt summary:** Provided draft paragraphs written by the student and requested sentence-level language editing: "Fix the grammar and improve the clarity of this paragraph without changing its meaning, technical terminology, or numerical results."

- **AI contribution:** Suggested corrections to grammar, sentence structure, word choice, and transitions. Proposed clearer wording for descriptions of the Linear, MLP, CNN, LSTM, and Transformer results. The requested scope was language editing of the supplied text, with the original interpretations and numerical values preserved.

- **Student verification:** Compared the suggested edits with the original student-written paragraphs. Checked that model names, technical terminology, numerical values, units, and comparisons remained consistent with the experimental results.

- **Affected files/sections:** Report, Readme.md, assignment1.html

- **Responsible member:** Hoàng Xuân Bách, Lê Nguyễn Gia Phúc, Nguyễn Việt Hùng.

## AI-Assisted Github Pages Skeleton

- **Tool:** GPT-5.6 Sol.

- **Development stage**: Planning (Milestone 1)

- **Used by:** Hoàng Xuân Bách, Lê Nguyễn Gia Phúc, Nguyễn Việt Hùng.

- **Task:** Creating the initial Github pages skeleton.

- **Prompt summary:** Requested a standard landing page layout and a basic HTML/CSS skeleton with placeholders.

- **AI contribution:** Assisted with creating the initial landing page skeleton, including basic HTML structure and layout suggestions.

- **Student verification:** Group members reviewed and refactored the generated HTML skeleton to align with the project specifications. Manually populated project-specific content, including team member profiles, course details, and assignment descriptions. Verified the semantic structure, navigation links, and responsive layouts across different screen sizes.

- **Affected files/sections:** index.html, style.css, assignment1.html, assignment2.html, assignment3.html.

- **Responsible members:** Hoàng Xuân Bách, Lê Nguyễn Gia Phúc, Nguyễn Việt Hùng.

## Consistency Verification Between Experimental Outputs and Report

- **Tool:** GPT-5.6 Sol.

- **Development stage**: Final Revision (Milestone 2)

- **Used by:** Lê Nguyễn Gia Phúc.

- **Task:** Supporting the final consistency check between experimental outputs and the report's implementation results.

- **Prompt summary:** Provided the final experimental outputs and the corresponding report content. Requested assistance with checking numerical consistency and identifying outdated values, mismatched labels, or explanations that were not supported by the results.

- **AI contribution:** Assisted with reviewing metric values, mean and sample standard deviation, parameter counts, computational costs, and Baseline/NoDropout comparisons. Suggested checking that tables, figures, captions, and conclusions referred to the same experimental configurations and aggregation conventions.

- **Student verification:** Compared the report's numerical statements, tables, figures, and captions with the final notebook outputs. Checked seed settings, variant labels, units, and aggregation conventions. Corrected outdated or inconsistent content and reviewed the conclusions against the experimental evidence.

- **Affected files/sections:** Report(Implementation Results)

- **Responsible member:** Lê Nguyễn Gia Phúc.

## Consistency Verification Between Experimental Outputs and GitHub Pages

- **Tool:** GPT-5.6 Sol.

- **Development stage**: Final Revision (Milestone 2)

- **Used by:** Hoàng Xuân Bách.

- **Task:** Supporting the final consistency check between experimental outputs and the results presented on GitHub Pages.

- **Prompt summary:** Provided the final experimental outputs and the corresponding GitHub Pages content. Requested assistance with checking whether the published numerical summaries, tables, figures, and explanations accurately represented the completed experiments.

- **AI contribution:** Assisted with reviewing the consistency of displayed metrics, model names, variant labels, units, and figure descriptions. Suggested checking that published figures and summaries used the final results, with single-seed examples distinguished from results aggregated across three seeds.

- **Student verification:** Compared the Assignment 1 results displayed on GitHub Pages with the final notebook outputs. Checked numerical summaries, table entries, figure versions, captions, and model labels. Corrected outdated content and inspected the rendered pages to confirm that the updated results and figures were displayed correctly.

- **Affected files/sections:** assignment1.html

- **Responsible member:** Hoàng Xuân Bách.

## LaTeX Formatting Assistance

- **Tool:** GPT-5.6 Sol.

- **Used by:** Lê Nguyễn Gia Phúc, Hoàng Xuân Bách, Nguyễn Việt Hùng

- **Development stage:** Report formatting and final revision (Milestone 2).

- **Task:** Improving the LaTeX presentation of student-provided report content.

- **Prompt summary:** Provided existing text, tables, and equations and requested formatting assistance. Requests included converting the experimental design into a comparison table, adjusting column widths and text alignment, splitting long tables, and formatting mathematical expressions without changing their intended meaning or numerical values.

- **AI contribution:** Suggested LaTeX environments and formatting code for tables, equations, and structured lists. Helped arrange the main comparison and dropout ablation side by side, merge shared experimental settings, and improve the readability of lengthy tables. Provided revised LaTeX snippets based on the supplied content.

- **Student verification:** Compiled the revised LaTeX and inspected the rendered report. Compared the formatted content with the original text and experimental outputs. Checked table entries, mathematical symbols, captions, labels, and cross-references, and adjusted spacing and page breaks where necessary.

- **Affected files/sections:** Report LaTeX source, particularly the experimental design, result tables, architecture equations, and AI-use disclosure sections.

- **Responsible member:** Lê Nguyễn Gia Phúc, Hoàng Xuân Bách, Nguyễn Việt Hùng
