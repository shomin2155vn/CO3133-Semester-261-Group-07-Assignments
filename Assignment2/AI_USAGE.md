# AI Usage Disclosure

<a id="sec:ai_usage"></a>

The group takes full responsibility for all submitted code, numerical results, figures, claims, and text. AI tools supported learning, reviewing and refining member-written code and drafts, building skeleton pages, and language editing. Group members reviewed the adopted suggestions and verified them against the implementation, experimental outputs, and assignment requirements. All reported experimental results were obtained from runs executed by the group, and result figures were produced from those outputs. AI assistance with code or presentation did not replace the execution of experiments or provide fabricated data or results. The following records describe the scope of AI assistance and the corresponding student verification.

## Review and Refinement of the Training and Evaluation Pipeline

- **Tool:** GPT-6.1 Sol.

- **Used by:** Lê Nguyễn Gia Phúc.

- **Development stage:** Pipeline review, refinement, and debugging (Milestone 1).

- **Task:** Review the member-written Food-11 classification pipeline for SimpleCNN, frozen ResNet-18, and fully fine-tuned ResNet-18. Document how to reconstruct the cleaned dataset splits, evaluate saved checkpoints, and reproduce the reported results.

- **Prompt summary:** Provided an existing member-written pipeline and specified the required workflow, current implementation of EDA, train/validation/test splitting, preprocessing and augmentation, model training pipeline, evaluation metrics, checkpoint selection, and reproducibility. Requested a review of the workflow, explanations of technical issues, and suggestions for improving reproducibility. Also requested appropriate ways to organize and save checkpoints, configurations, experimental outputs.

- **AI contribution:** Reviewed the supplied pipeline and suggested improvements to data preparation, training, evaluation, checkpoint selection, and reproducibility. Recommended saving model configurations, dataset split indices, seed settings, dependency versions, hardware information, checkpoints, metric summaries, and training histories. Helped modify visualization code for learning curves, confusion matrices, and prediction examples.

- **Student verification:** Compared the suggested changes with the original member-written pipeline and the experimental requirements. Reviewed the data split, preprocessing, training settings, metric calculations, checkpoint-selection rule, and reproducibility settings. Executed the training workflow and inspected the generated metrics and checkpoints.

- **Affected files/sections:** Assignment2/Assignment2.ipynb.

- **Responsible member:** Lê Nguyễn Gia Phúc.

## Model Architecture Review and Technical Verification

- **Tool:** GPT-6.1 Sol

- **Development stage:** Model Implementation (Milestone 1)

- **Used by:** Lê Nguyễn Gia Phúc

- **Task:** Review the Food-11 implementations of SimpleCNN, frozen ResNet-18, and fully fine-tuned ResNet-18. Check tensor dimensions, residual connections, classifier heads, and trainable parameter counts.

- **Prompt summary:** The member provided the model implementations and draft architecture descriptions. The prompts asked how the models process 3 × 256 × 256 RGB inputs, where downsampling occurs in ResNet-18, and how the shortcut and residual branches match in shape before addition.

- **AI contribution:** Checked the ResNet-18 description against the TorchVision source and explained the stem, the two convolutions in each residual branch, and the stride-2 projection shortcuts at the start of Stages 2, 3, and 4. Helping to revise the architecture description

- **Student verification:** The member compared their original ResNet-18 description with provided explanation and checked both against the official TorchVision documentation and source code

- **Affected files/sections:** Assignment2/Assignment2.ipynb, Report(Methodology)

- **Responsible member:** Lê Nguyễn Gia Phúc.

## Feedback on Results Interpretation and Discussion

- **Tool:** GPT-6.1 Sol

- **Development stage:** Report writing (Milestone 3)

- **Used by:** Nguyễn Việt Hùng, Hoàng Xuân Bách, Lê Nguyễn Gia Phúc.

- **Task:** Reviewing the student’s initial interpretations and suggesting improvements to the organization and focus of the comparative discussion.

- **Prompt summary:** Provided the assignment’s discussion requirements, the experimental results already obtained, and an initial student-written draft explaining those results. Requested feedback on the draft and suggestions on how to organize and refine the explanations in the report, linking relevant comparisons, observed patterns, and supporting evidence to each requirement. The aim was to keep the discussion focused on the specification and available evidence, while avoiding repetitive descriptions, irrelevant commentary, and unsupported conclusions.

- **AI contribution:** Reviewed the supplied draft and suggested revisions to its structure, wording, and interpretation of the results. Helped connect overall performance, class-specific differences, confusion patterns, training behavior, and computational trade-offs to the assignment requirements. Suggested qualifying explanations related to architecture, regularization, and training where the experiments did not isolate their individual effects.

- **Student verification:** Compared the suggested revisions with the original draft and the assignment’s discussion requirements. Checked numerical comparisons against notebook outputs and result tables, and assessed interpretations against the learning curves, confusion matrices, and prediction examples.

- **Affected files/sections:** Report(Implementation Results)

- **Responsible member:** Nguyễn Việt Hùng, Hoàng Xuân Bách, Lê Nguyễn Gia Phúc.

## English Language Editing

- **Tool:** GPT-6.1 Sol.

- **Development stage:** Final Revision (Milestone 3)

- **Used by:** Hoàng Xuân Bách,Lê Nguyễn Gia Phúc, Nguyễn Việt Hùng

- **Task:** Improving the grammar, clarity, and readability of student-written explanations in the report.

- **Prompt summary:** Provided draft paragraphs written by the student and requested sentence-level language editing: "Fix the grammar and improve the clarity of this paragraph without changing its meaning, technical terminology, or numerical results."

- **AI contribution:** Suggested corrections to grammar, sentence structure, word choice, and transitions. Proposed clearer wording for descriptions of the SimpleCNN, frozen Resnet18 and Resnet18 full fine-tuning results. The requested scope was language editing of the supplied text, with the original interpretations and numerical values preserved.

- **Student verification:** Compared the suggested edits with the original student-written paragraphs. Checked that model names, technical terminology, numerical values, units, and comparisons remained consistent with the experimental results.

- **Affected files/sections:** Report, Readme.md, assignment2.html

- **Responsible member:** Hoàng Xuân Bách,Lê Nguyễn Gia Phúc, Nguyễn Việt Hùng

## AI-Assisted Github Pages Skeleton

- **Tool:** GPT-5.6 Sol.

- **Development stage:** Planning (Milestone 1)

- **Used by:** Hoàng Xuân Bách, Lê Nguyễn Gia Phúc, Nguyễn Việt Hùng.

- **Task:** Create the initial HTML/CSS skeleton for the group website, including a landing page, assignment pages, and reusable table styles.

- **Prompt summary:** Requested a landing page layout and a basic HTML/CSS skeleton with placeholder sections, sample table markup, and shared styles for table headers, cells, and captions.

- **AI contribution:** The tool suggested the initial page structure and navigation layout. It also provided sample table markup and reusable CSS for consistent table formatting across the assignment pages.

- **Student verification:** Reviewed and revised the generated skeleton against the project requirements. Members added the course details, team profiles, and assignment descriptions. Also checked the semantic HTML structure, navigation links, table headers and captions, and how the layouts displayed at different screen sizes.

- **Affected files/sections:** index.html, style.css, assignment1.html, assignment2.html, assignment3.html.

- **Responsible members:** Hoàng Xuân Bách, Lê Nguyễn Gia Phúc, Nguyễn Việt Hùng.

## Consistency Verification Between Experimental Outputs and Report

- **Tool:** GPT-6.1 Sol.

- **Development stage:** Final Revision (Milestone 3)

- **Used by:** Lê Nguyễn Gia Phúc.

- **Task:** Supporting the final consistency check between experimental outputs and the report’s implementation results.

- **Prompt summary:** Provided the final experimental outputs and the corresponding report content. Requested assistance with checking numerical consistency and identifying outdated values, mismatched labels, or explanations that were not supported by the results.

- **AI contribution:** Assisted with reviewing metric values, mean and sample standard deviation, parameter counts, computational costs, and Baseline/NoAugmentation comparisons. Suggested checking that tables, figures, captions, and conclusions referred to the same experimental configurations and aggregation conventions.

- **Student verification:** Compared the report’s numerical statements, tables, figures, and captions with the final notebook outputs. Checked seed settings, variant labels, units, and aggregation conventions. Corrected outdated or inconsistent content and reviewed the conclusions against the experimental evidence.

- **Affected files/sections:** Report(Implementation Results)

- **Responsible member:** Lê Nguyễn Gia Phúc.

## Consistency Verification Between Experimental Outputs and GitHub Pages

- **Tool:** GPT-6.1 Sol.

- **Development stage:** Final Revision (Milestone 3)

- **Used by:** Hoàng Xuân Bách.

- **Task:** Supporting the final consistency check between experimental outputs and the results presented on GitHub Pages.

- **Prompt summary:** Provided the final experimental outputs and the corresponding GitHub Pages content. Requested assistance with checking whether the published numerical summaries, tables, figures, and explanations accurately represented the completed experiments.

- **AI contribution:** Assisted with reviewing the consistency of displayed metrics, model names, variant labels, units, and figure descriptions. Suggested checking that published figures and summaries used the final results, with single-seed examples distinguished from results aggregated across three seeds.

- **Student verification:** Compared the Assignment 2 results displayed on GitHub Pages with the final notebook outputs. Checked numerical summaries, table entries, figure versions, captions, and model labels. Corrected outdated content and inspected the rendered pages to confirm that the updated results and figures were displayed correctly.

- **Affected files/sections:** assignment2.html

- **Responsible member:** Hoàng Xuân Bách.

## LaTeX Formatting Assistance

- **Tool:** GPT-6.1 Sol.

- **Used by:** Lê Nguyễn Gia Phúc, Hoàng Xuân Bách, Nguyễn Việt Hùng

- **Development stage:** Report formatting and final revision (Milestone 3).

- **Task:** Improving the LaTeX presentation of student-provided report content.

- **Prompt summary:** Provided existing text, tables, and equations and requested formatting assistance. Requests included converting the experimental design into a comparison table, adjusting column widths and text alignment, splitting long tables, and formatting mathematical expressions without changing their intended meaning or numerical values.

- **AI contribution:** Suggested LaTeX environments and formatting code for tables, equations, and structured lists. Helped arrange the main comparison and ablation study side by side, merge shared experimental settings, and improve the readability of lengthy tables. Provided revised LaTeX snippets based on the supplied content.

- **Student verification:** Compiled the revised LaTeX and inspected the rendered report. Compared the formatted content with the original text and experimental outputs. Checked table entries, mathematical symbols, captions, labels, and cross-references, and adjusted spacing and page breaks where necessary.

- **Affected files/sections:** Report(LaTeX source code)

- **Responsible member:** Lê Nguyễn Gia Phúc, Hoàng Xuân Bách, Nguyễn Việt Hùng
