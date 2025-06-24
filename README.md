# BugDetectiveAI
Using LLMs to act as a Detective for buggy code, and evaluating the results over multiple metrics.

# Dataset Description:
## Buggy dataset:
The buggy dataset includes three pickle-files, containing buggy snippets from training, validation and test sets as well as a folder with source code of all its snippets. Each training or validation pickle-file contains a pandas dataframe with the following columns:

`before_merge` - implementation of a snippet, containing a bug;
`after_merge` - implementation of a snippet, being an immediate fix of bugs in the corresponding snippet in the column before_merge;
`filename` - filename where buggy and its corresponding fixed snippets reside;
`full_file_code_before_merge` - source code of the module, containing the buggy snippet;
`full_file_code_after_merge` - source code of the module, containing the fixed snippet;
`function_name` - a complete function/method name;
`url` - issue url, where bugs are reported in the buggy snippet;
`source code and errors` - contains parsed source code and error messages from the issue report;
`full_traceback` - complete traceback report;
`traceback_type` - exception type in the traceback report;
`before_merge_without_docstrings` - implementation of the buggy snippet without its comments and docstrings;
`after_merge_without_docstrings` - implementation of the fixed snippet without its comments and docstrings;
`before_merge_docstrings` - docstrings in the buggy snippet;
`after_merge_docstrings` - docstrings in the fixed snippet;
`path_to_snippet_before_merge` - path to the file with source code of the buggy snippet;
`path_to_snippet_after_merge` - path to the file with source code of the fixed snippet.
Its test pickle-file contains a table with the following columns:

`before_merge` - as above;
`after_merge` - as above;
`url` - as above
`bug type` - a type of a bug ib the snippet according to the known classification of bugs from https://cwe.mitre.org/
`bug description` - textual description of the bug;
`bug filename` - a filename where the buggy snippet resides;
`bug function_name` - a complete function/method name;
`bug lines` - lines ranges in the snippet source code where the bug is supposed to be located;
`full_traceback` - as above;
`traceback_type` - as above;
`path_to_snippet_before_merge` - as above;
`path_to_snippet_after_merge` - as above.


#Metrics:
- Exact Match (EM)
- String-Based Similarity: CodeBleu
- AST-Based Similarity
- Diff Based metrics
