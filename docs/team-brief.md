# M112 · TimbreLab

**Team implementation brief for Charaf, Chems, Basma and Layla**

Share this whole file with each teammate and their AI. Read sections 1–3 together. Each person can then start their block in section 4. Sections 5–6 explain how to combine the work on GitHub and how to decide the project is finished.

## 1. What we are building and how it will work

We are building a local Python application for exploring data and comparing machine learning models. Music is our main example. A user can listen to a clip, see its waveform and see which genre the model predicts. The same app also accepts ordinary CSV tables so users can predict a numerical value or explore groups in their own data.

| What the user wants to do | What happens in the app | What the user gets |
|---|---|---|
| **Explore data** | Open the included example, upload a CSV, or create a feature table from labelled WAV clips. Inspect rows, missing values, distributions and the number of clips per genre. | A preview, charts and a downloadable table. |
| **Classify music** | Train candidate classifiers on labelled clips. Compare them using the same recordings and features. Select a trained model and upload a new WAV clip. | Predicted genre, comparison scores and a chart showing which genres are confused. |
| **Run a regression** | Import a table with a numeric target. Select one predictor for simple or polynomial regression, or several for multiple regression. | Predicted values, error scores and an actual-versus-predicted plot. |
| **Discover groups** | Select numeric features and a number of clusters. K-means groups similar rows without using the genre label. | A cluster number per row, a plot of the groups and a description of what distinguishes them. |
| **Save work** | Download the results of the current run. | Predictions as CSV, charts as PNG and the fitted model with its preprocessing and feature information. |

**Example presentation flow.** Play a short clip. Show how its tempo, energy and brightness become one row in a table. Run two classifiers on the same data and examine one genre they confuse. Predict a genre for a new WAV file. Then use clustering on the music table and regression on a small numeric CSV. Export one result in front of the audience.

**Course scope.** The planned model list is simple, polynomial and multiple linear regression, logistic regression, decision tree, Gaussian Naive Bayes, SVM, K-nearest neighbours, Random Forest, a small feed-forward neural network and K-means. Logistic regression is used for classification. The brief asks for implementation but does not specify how much must be coded from scratch. Confirm the instructor’s expectation early and record it in the README. This plan assumes scikit-learn is allowed. If a model must be written by hand, change its internal implementation and keep the shared interface below.

**Recommended stack.** Use Streamlit for the interface, pandas and NumPy for tables, librosa and soundfile for audio, scikit-learn for models, Matplotlib for figures and joblib for model export. The core demo should run locally after setup with one command, `streamlit run app.py`. Use an included, shareable example CSV so the audience can run it without downloading a music collection.

## 2. Agree on the connections before splitting up

The part everyone depends on is the **shape of the data**, not Charaf’s finished audio code. Make a short first pull request with the conventions below and small example files. Everyone can then build against those examples.

### Audio table

One row represents one clip. The canonical training table uses the following columns.

| Column | Meaning | Use |
|---|---|---|
| `clip_id` | Unique text ID for this clip | Trace a result to its source. Never a predictor. |
| `recording_id` | ID of the original recording; shared across excerpts from it | Keep excerpts together when splitting data. Never a predictor. |
| `genre` | Known genre for a training clip | Classification target. A new clip has no known genre. |
| `tempo_bpm` | Estimated beats per minute | Numeric predictor. Leave missing if no usable estimate is available. |
| `rms_mean` | Average signal energy measure | Numeric predictor. |
| `centroid_hz_mean` | Average spectral centroid, a measure related to brightness | Numeric predictor. |
| `mfcc_01_mean` through `mfcc_13_mean` | Thirteen measurements describing timbre, or sound character | Thirteen separate numeric predictors. |

Charaf owns one ordered list of these **16 predictors** in the audio module. Training and prediction both use it. The table and saved audio model carry the same feature version, initially `audio_v1`. Changing a column name, unit or calculation requires updating the version, fixture and model together. File names, folder names, IDs and target labels must never enter the audio predictor matrix.

**Audio convention.** Start with WAV input, mono audio resampled to 22,050 Hz, and up to the first 30 seconds. Use the available audio for shorter clips. Use the same extraction function and settings for the dataset and a new upload. Report unreadable, empty, silent or unusably short clips by name. A batch should still process the valid clips. For the labelled demo, use GTZAN with a documented download and preparation procedure. Keep original recording IDs when cutting clips, and identify or document known duplicate recordings before evaluation.

**Ordinary CSV convention.** The user picks the target and the predictor columns. Support numeric predictors in the first version and allow text category labels such as genres. Show which columns are excluded and why. The target and identifier columns must be excluded from predictors. Rows without a supervised target cannot be used for supervised training. For regression, supply a small example with a meaningful numeric target, such as predicting a restaurant tip from the bill and party size.

### What the modules exchange

| Owner | Shared entry point | What the caller can rely on |
|---|---|---|
| Charaf | `extract_audio_features(wav_path)` | Returns one dictionary containing the 16 named numeric features. Dataset creation adds IDs and genre separately. |
| Basma | `train_supervised(table, task, target, features, model_name, group_column=None)` | Accepts a DataFrame and selected column names. Returns the result described below. Works for classification and regression, including Charaf’s Random Forest. |
| Layla | `run_clustering(table, features, k)` | Returns the same result format with cluster assignments and a fitted clustering pipeline. |
| Chems | App and export layer | Calls the functions above, displays their results and exports them. Owns upload handling, temporary files and user-facing error messages. |

Use a plain dictionary for each model result. Give it four agreed keys: `pipeline` for the fitted preprocessing and model, `metrics` for named scores, `predictions` for a DataFrame, and `metadata` for the task, model name, ordered feature names and settings. Audio model metadata also includes the feature version. For supervised runs, the prediction table contains `row_id`, `actual` and `predicted` for held-out rows. For clustering it contains `row_id` and `cluster` for every analysed row. Use the clip ID as the row ID where available.

**New WAV prediction.** Enable this action only for a genre classifier trained on the agreed audio feature schema. Extract the features, put them in the saved order and call the fitted pipeline’s prediction method. Export the pipeline and metadata together. A model trained on an unrelated table, such as restaurant tips, should only offer predictions for matching tabular inputs.

## 3. First shared checkpoint

1. **Chems** creates the shared GitHub repository and the small folder structure below. The four members agree on names, inputs and outputs in one short meeting.
2. **Basma and Charaf** provide a tiny audio-shaped CSV with invented values and a numeric regression CSV. Include multiple rows and classes. Mark both as examples. Invented audio values are for checking connections only.
3. **Layla** opens the example table and verifies that the agreed IDs, labels and numeric columns are enough for her charts. **Chems** shows the same rows in the app.
4. Merge this foundation into `main`. No one needs a trained model to begin. As soon as Charaf has **one real feature row**, check that it loads without renaming columns. Train, compare and report final music results only when the real labelled dataset is ready.

~~~text
app.py                  Chems · interface and downloads
audio/                  Charaf · extraction, dataset creation, Random Forest
models/                 Basma · common training, preprocessing, evaluation
analysis/               Layla · clustering and figures
data/examples/          Small shareable fixtures and demo tables
tests/                  Shared checks for the connections between modules
docs/                   Tutorial, report sections and presentation notes
README.md               Setup, run instructions and team contributions
requirements.txt        Agreed dependencies for the tested environment
~~~

## 4. Four task blocks to work on simultaneously

### Charaf · Audio features and the genre classifier

**Your outcome:** someone can give the app a new WAV clip and see a genre predicted from its sound.

1. Implement the audio convention and feature extractor from section 2. Keep signal processing separate from the interface. Provide one short clip the team has permission to share and a real row extracted from it.
2. Build a repeatable script that reads labelled recordings and writes the feature CSV. Document the source, label mapping, skipped files, duplicates and final counts. Keep a record of the extraction settings. Cache the CSV so teammates can work without repeatedly processing audio.
3. Provide the Random Forest estimator in `audio/`. Give Basma the factory function or estimator to register as `random_forest` in the shared trainer. Use Basma’s evaluation workflow so your scores can be compared with the other classifiers.
4. With Chems, check that the same WAV produces the same features through dataset creation and through upload. Prepare one independent demo clip and explain one correct prediction and one mistake from held-out data.
5. Write the report section on the dataset, features, Random Forest and limitations. Include a labelled waveform or spectrogram with the measurement units.

**You can start with:** a single WAV. **Your first handoff:** one real feature row for Chems and the column list for Basma and Layla. **Done when:** another member can recreate the table from the documented input and predict a genre through the app using the saved model.

### Chems · Interface, integration and delivery

**Your outcome:** a teammate can install the submitted project, run the demo and save the results.

1. Build a straightforward flow with **Load data → Explore → Choose task and model → Results → Export**. Start with the example tables. Offer CSV import, labelled WAV batch import or preparation through Charaf’s documented script, and single-WAV prediction.
2. Display the dataset size, selected columns, missing values and any skipped rows or files. Show only models that support the selected task. Explain invalid choices in plain language, such as “Choose a numeric target for regression.”
3. Connect Basma’s trainer and Layla’s clustering function. During early development, use a clearly marked sample result to build result screens. Replace it with the real function before merging the finished feature.
4. Add CSV prediction downloads, PNG chart downloads and fitted-model export with metadata. Clear previous results when the dataset, task, feature selection or model changes so an old score cannot appear under a new selection.
5. Maintain the README and dependencies. Build a clean source ZIP and test it after extracting it into a new folder on a second laptop. Include example data, setup instructions and the demo run command.

**You can start with:** mock CSVs and the result format in section 2. **Your first handoff:** teammates can open the app and inspect their example rows. **Done when:** another member completes every demo workflow from the ZIP and opens the downloaded results.

### Basma · Data preparation, supervised models and evaluation

**Your outcome:** the team can compare models fairly, explain the scores and repeat the experiments.

1. Implement the common trainer. Validate the target, selected numeric features and available training examples. Split data **before** learning imputation or scaling. Put preprocessing and the estimator in a single fitted pipeline.
2. For classification, implement logistic regression, decision tree, Gaussian Naive Bayes, SVM, K-nearest neighbours and a small feed-forward neural network. Register Charaf’s Random Forest. For regression, implement simple linear, multiple linear and polynomial regression. Make the number of predictors and polynomial degree explicit.
3. Use one reproducible split for all models compared on the same task. Split audio by original recording, keeping known duplicates or excerpts of the same recording on one side. Check class coverage. If the sample is too small for a meaningful split, explain why. Tune or compare candidates on training/validation data and reserve the final test set for reporting.
4. For classification, return accuracy and macro F1 (each class has equal weight), plus actual and predicted labels for the confusion matrix. For regression, return mean absolute error in the target’s units and R² when defined. Compare the best candidate against a simple reference, such as always predicting the most frequent class or the training mean.
5. Give Layla the prediction tables and scores. Write the report section on preprocessing, algorithms, evaluation and errors. Record the settings and split so the numbers in the report can be reproduced.

**You can start with:** example tables, even before the audio extractor works. **Your first handoff:** one classifier and one regression model returning the agreed result format. **Done when:** all required supervised models run on suitable inputs, reject unsuitable inputs clearly, and produce reproducible results on real data.

### Layla · Clustering, charts and report

**Your outcome:** users can explore similar clips, understand the results visually and follow a complete report.

1. Build dataset charts for class counts, missing values and feature distributions. Label axes and units. Use the team’s shared examples until Charaf’s real table is ready.
2. Implement K-means using numeric features with missing-value handling and scaling. Exclude IDs and genre from clustering. Let the user choose a valid number of groups. Return one cluster per analysed row. Produce a readable two-dimensional view and label the axes with the actual features or projection used.
3. Describe clusters using feature summaries, such as average tempo or brightness. If genre labels exist, compare them with clusters afterward. Report a silhouette score only when the assignments contain enough distinct groups for it to be defined.
4. Build the classification confusion matrix and regression actual-versus-predicted plot from Basma’s result tables. Provide chart functions that Chems can both display and export.
5. Create the report outline early. Each member writes their own technical section. Assemble the final PDF with the problem, data source, architecture, preparation, implemented algorithms, evaluation, results, limitations, short user tutorial, references and member contributions. Use screenshots and scores from the final running version.
6. Assemble a 20-minute group presentation and organise the printed report. Give each member time to explain or demonstrate the part they implemented.

**You can start with:** the example CSVs and a small invented prediction table. **Your first handoff:** one cluster result and one exportable chart. **Done when:** the visuals use real final results, the report agrees with the app, and another member can follow the tutorial without help.

## 5. How we use GitHub without blocking one another

1. **One issue per deliverable.** Give each issue an owner, the input it expects, the output it produces and one observable check. Example: “Upload a WAV, get a row with the 16 agreed predictors, and show a useful message for an empty file.”
2. **Branch from current `main` for one small task.** Example branch names include `audio/extract-features`, `ui/csv-preview`, `models/regression` and `analysis/kmeans`. Open small pull requests as pieces work. Avoid keeping a whole member’s work on one branch until the end.
3. **One teammate reviews each pull request.** The author lists what changed, how to run it and the observed result. The teammate who will consume the output should review the first connection. For example, Chems reviews Charaf’s first feature row and Layla reviews Basma’s first prediction table.
4. **Keep shared files predictable.** Chems maintains `app.py` and the dependency list. Each owner works mainly in their folder. If a shared column or return value must change, agree in the issue first and update producers, consumers, example data and checks in the same pull request.
5. **Integrate one complete path early.** Run one real WAV through extraction, load the resulting table in the app, train one classifier on enough real recordings and export its predictions. Repeat with one regression and one clustering run before spending time on appearance or extra models.
6. **Keep the repository small and reproducible.** Commit source, small shareable fixtures, setup instructions and the data preparation script. Keep the full music collection, virtual environments, caches and temporary uploads out of Git. Document how to obtain the data. Include only audio that may be redistributed.
7. **Create one final release.** After the checks below pass, label the accepted GitHub commit and build the submission ZIP from it. Produce the PDF screenshots and final score tables from that same version.

**For teammates using AI:** give your AI this full brief, the relevant GitHub issue and the current repository files. Tell it which member you are. Ask it to work on one owned deliverable, follow the shared interfaces, run the listed check and explain the result so you can defend the implementation during the presentation. If it proposes changing another member’s interface, discuss the change in the issue before merging it.

## 6. Final checks and handover

| Check | Evidence that it passes |
|---|---|
| Clean setup | A teammate extracts the source ZIP on another laptop, follows the README, starts the app and opens the included demo. |
| Import and preparation | The app handles a valid CSV, missing feature values, an invalid target, a valid WAV and an unreadable WAV. Messages identify what the user needs to fix. |
| Music classification | A real labelled dataset trains successfully. A held-out recording receives a prediction. Recording IDs do not overlap between training and test sets. |
| Regression and clustering | A numeric example produces predictions and error metrics. K-means returns assignments and a chart without using the target label. |
| Exports | Exported CSV rows match the displayed run. PNG files open. Reloading a team-generated saved pipeline and its metadata reproduces a prediction on the same input. |
| Report and tutorial | Every reported result can be traced to the final code and dataset. Every member has reviewed their section. Someone else has followed the tutorial. |
| Submission and presentation | The group has the detailed PDF report and source ZIP ready for Google Classroom, a printed report for presentation day and a rehearsed 20-minute demonstration. |
