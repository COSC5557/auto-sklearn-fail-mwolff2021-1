[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11789046&assignment_repo_type=AssignmentRepo)
# Auto-sklearn Fail

The code in `fail.py` runs
[auto-sklearn](https://automl.github.io/auto-sklearn/master/) on a dataset for 5
minutes. The model it finds after that time is *worse* than just using a random
forest with default hyperparameters.

Find out what's going on, why auto-sklearn's performance is so bad, and how to
fix it.

## References
I discussed this problem with another student (Russell Todd) and we collaborated on different approaches to the problem, but did not share code with each other/pair program. 

Links: 

https://www.analyticsvidhya.com/blog/2020/07/10-techniques-to-deal-with-class-imbalance-in-machine-learning/
https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
https://datagy.io/python-zip-lists/
https://community.anaconda.cloud/t/cannot-import-name-onetoonefeaturemixin-from-sklearn-base/47735
https://stackoverflow.com/questions/56549270/importerror-cannot-import-name-multioutputmixin-from-sklearn-base
https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_get_pipeline_components.html#sphx-glr-examples-40-advanced-example-get-pipeline-components-py
https://automl.github.io/auto-sklearn/master/faq.html#faq
https://automl.github.io/auto-sklearn/master/manual.html#manual
https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_pandas_train_test.html#sphx-glr-examples-40-advanced-example-pandas-train-test-py
https://stackoverflow.com/questions/26097916/convert-pandas-series-to-dataframe
https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_get_pipeline_components.html#sphx-glr-examples-40-advanced-example-get-pipeline-components-py
https://github.com/automl/auto-sklearn/issues/633
https://stackoverflow.com/questions/56549270/importerror-cannot-import-name-multioutputmixin-from-sklearn-base
https://community.anaconda.cloud/t/cannot-import-name-onetoonefeaturemixin-from-sklearn-base/47735
https://automl.github.io/auto-sklearn/master/installation.html
https://adamnovotny.com/blog/google-colab-and-automl-auto-sklearn-setup.html
https://stackoverflow.com/questions/53839948/how-to-install-auto-sklearn-on-googlecolab
https://stackoverflow.com/questions/12555323/how-to-add-a-new-column-to-an-existing-dataframe
https://automl.github.io/auto-sklearn/master/examples/20_basic/example_classification.html
https://stackoverflow.com/questions/40155128/plot-trees-for-a-random-forest-in-python-with-scikit-learn
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
https://automl.github.io/auto-sklearn/master/examples/20_basic/example_classification.html#sphx-glr-examples-20-basic-example-classification-py
https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_multi_objective.html#sphx-glr-examples-40-advanced-example-multi-objective-py
https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_debug_logging.html#sphx-glr-examples-40-advanced-example-debug-logging-py
https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_pandas_train_test.html#sphx-glr-examples-40-advanced-example-pandas-train-test-py
https://automl.github.io/auto-sklearn/master/examples/index.html#examples
https://automl.github.io/auto-sklearn/master/faq.html#results-log-files-and-output
https://tex.stackexchange.com/questions/63981/references-not-printing
https://www.w3schools.com/python/ref_random_seed.asp
https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
https://automl.github.io/auto-sklearn/master/examples/40_advanced/example_debug_logging.html
