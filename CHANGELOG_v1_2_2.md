# EvalWeave v1.2.2

- Added configuration-only templates for domain multiple-choice and text-generation tasks.
- Added dynamic custom task loading through `type: custom` and `class_path`, without editing the task factory.
- Added a keyword-coverage scorer as a complete custom-plugin example.
- Added `validate-task` for checking every sample before running a model.
- Resolved relative dataset paths against the task configuration file.
- Added official-source preparation script for rebuilding the full public suite after cloning.
- Added a plain-language domain-task guide and GitHub-ready README.
