# Training results overview

_Generated 2026-11-08 from `results/predictions/` (the models' raw outputs). **Provisional:** the full statistics (confidence intervals, significance tests, calibration, explanations, network analysis, speed tests) come from `bash run_all.sh analysis` and will replace these numbers. Values are mean ± SD (standard deviation) over runs unless one single run is shown._

## Start here: the idea in plain words

- **The task.** Give the computer a photo of a fingernail. It must say which one of 7 things it sees (6 nail conditions or a healthy nail).
- **Why we trained everything again.** The reviewer of our paper pointed out a problem: in the original experiment the same photo, or near-identical copies of it, could end up in both the *training* part (what the model learns from) and the *test* part (what we use to grade it). Then the model is graded on pictures it has effectively already seen, and the scores look much better than they really are. We therefore rebuilt the data split properly and retrained all five models many times.
- **What this file is.** A summary of all 180 training runs, section by section. Each section says what that training does, why we ran it, which classes it used, and then shows the results.
- **Where to look first.** The *Key takeaways* section (after the class tables) gives the short version.

## Words used in this file

| Word | Meaning |
|---|---|
| Class | One of the 7 labels the model chooses from (for example `onychomycosis`). |
| Training / validation / test | Training photos are what the model learns from. Validation photos are used while training to decide when to stop. Test photos are only used at the end, to give the final grade. |
| Near-duplicate group ("component") | One original photo plus every copy of it (rotated, sheared, re-saved or re-uploaded). We keep each group entirely inside train, validation or test, never split across them. |
| Fold / repeat | We cut the groups into 5 parts (folds). Each part takes a turn as the test set. We do the whole thing 3 times with different random cuts (repeats), giving 15 runs per model so we can see how stable the result is. |
| Epoch / early stopping | One epoch = one pass of learning. Early stopping = stop when the validation score has not improved for 6 epochs and keep the best version. |
| Accuracy | Share of photos classified correctly. Can look good even if a rare class is always wrong. |
| Balanced accuracy | Average of the share found in each class. Treats every class equally. |
| Macro-F1 | A score from 0 to 1 that balances *missing* cases and *false alarms* in each class, then averages over classes. 1.0 is perfect. It is our main score because it does not let big classes hide mistakes in small ones. Only classes that exist in the test set are counted. |
| Recall (per class) | Of the photos that really are class X, the share the model finds. |
| External test set | Photos from a different source than the training data. This shows whether the model works outside the dataset it learned from. |
| Domain shift | The drop in performance when photos come from a different camera, hospital or website than the training photos. |
| Checkpoint | The saved trained model (`best.pt`). All 180 are kept in `runs/`. |
| ± (plus-minus) | Mean ± standard deviation across runs. A smaller second number means a more stable result. |

## The 7 classes (what we trained on)

All main trainings use the same 7 classes. The short descriptions are plain-language summaries for readers of this file, not medical advice, and no dermatologist has checked the labels.

| Class | Name | In plain words | ko-pin groups / images used in cv |
|---|---|---|---:|
| `beau_s` | Beau's lines | Horizontal grooves or dents across the nail, like a small ditch running from side to side. They can appear when nail growth is briefly interrupted (for example after illness or an injury). | 142 / 1,492 |
| `black_line` | Black line | A dark brown or black stripe running along the length of the nail. Often harmless, but in rare cases it can be a sign of a skin cancer (melanoma). **Safety note:** a model must never be used to decide this on its own. | 227 / 5,450 |
| `clubbing` | Clubbing | The fingertip becomes wider and rounder and the nail curves strongly downwards, like an upside-down spoon. It can go together with long-term heart or lung problems. | 194 / 6,074 |
| `healthy_nail` | Healthy nail | A normal nail with none of the signs above. | 284 / 5,287 |
| `onychomycosis` | Onychomycosis | A fungal infection of the nail. The nail usually becomes thick, yellow or brown, and crumbly. | 645 / 4,241 |
| `onycholysis` | Onycholysis | The nail lifts and separates from the skin under it, usually starting at the tip, leaving a white or cloudy gap. | 743 / 5,090 |
| `white_spot` | White spot | Small white marks on the nail plate (medical name: leukonychia). | 301 / 5,464 |

The last column counts what is available for training and testing after cleaning ko-pin: groups of near-identical photos and the number of image files in them. Some classes are small in *distinct photos* (for example Beau's lines: 142 groups) even though they have many files, because the files are mostly copies of the same few photos.

## Which dataset has which class

We use several datasets. **Not every dataset has all 7 classes**, and this matters for every result below. The counts are all labelled images of that dataset, copies included.

| Dataset | Images | Classes present | beau_s | black_line | clubbing | healthy_nail | onychomycosis | onycholysis | white_spot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kopin | 43,144 | 7 | 6,135 | 6,105 | 6,320 | 5,972 | 6,304 | 6,065 | 6,243 |
| han (hospital) | 31,437 | 5 | – | 2,649 | – | 18,116 | 7,614 | 2,716 | 342 |
| han_virtualE | 1,732 | 4 | – | 185 | – | 424 | 760 | 363 | – |
| thai (3 Kaggle sets) | 1,040 | 6 | 220 | 220 | 160 | 60 | – | 220 | 160 |
| nikhilgurav21 | 1,126 | 2 | – | – | 783 | 343 | – | – | – |
| kaggle17 | 201 | 4 | 50 | – | 52 | – | – | 62 | 37 |
| josephrasanjana | 1,032 | 2 | – | – | – | 310 | 722 | – | – |
| farisfaikar | 293 | 1 | – | – | – | 293 | – | – | – |
| isratjahanmitu | 556 | 1 | – | – | – | – | 556 | – | – |

In short: ko-pin is the only dataset with all 7 classes. The Han hospital data has 5 (no Beau's lines, no clubbing). nikhilgurav21 and josephrasanjana have only 2 classes each, and farisfaikar and isratjahanmitu only 1. A model can only learn to name classes it has seen in training.

## Key takeaways

- **Main result (cv):** macro-F1 is 0.883–0.913 across the five models. The best two, EfficientNet-B2 (0.913) and ViT-B/16 (0.911), differ by less than the normal run-to-run wobble (about 0.019), so this data cannot separate them. Formal tests come later (stage s07).
- **Hardest classes:** beau_s (average recall 0.76) and white_spot (0.83). onychomycosis and healthy_nail are found at least 96% of the time.
- **Low-quality images (~100 px):** ViT-B/16 keeps macro-F1 0.86; the four CNNs fall to 0.66–0.73.
- **Leakage is large:** the original-style random split scores 8.1–11.5 points higher than the careful split (almost 1.00 instead of about 0.9). That gain comes from near-identical copies sitting in both train and test, not from a better model.
- **Other datasets are much harder:** models trained on ko-pin reach only 0.37–0.42 macro-F1 on the Han hospital images (17,697 images, 5 classes), against 0.88–0.91 on ko-pin itself.
- **Train on one dataset, test on another (xsrc):** scores on the training dataset's own test images average 0.90, but only 0.30 on other datasets. ko-pin-trained models average 0.61 on other datasets and models trained on the other datasets average 0.25. Part of this is class coverage: ko-pin is the only dataset with all 7 classes, and nikhilgurav21 and josephrasanjana have only 2 classes each (a 2-class model cannot name the other classes, and its own 2-class test is easy).
- **Pooling the other datasets (msrc):** scores on the held-out group are low mostly because that group contains classes the pool barely has (for example, no onychomycosis when the Han hospital data is held out). Even with nothing held out, the pooled models reach only 0.43 on ko-pin.
- **Short training (5 epochs):** costs the four CNNs 2.5–4.7 points, but ViT-B/16 changes by +2.6 points.

## 1. What was trained

**What it does.** We took five well-known image-recognition models that were already trained on millions of everyday photos (ImageNet) and re-trained them to recognise nail classes. Starting from a model that already knows general shapes and textures is called *transfer learning*; it works much better than starting from nothing when the nail dataset is small.

**Why these five.** They cover very different designs, so our conclusions do not depend on one model: old and simple (VGG16), feature-reusing (DenseNet121), balanced (EfficientNet-B2), tiny and phone-ready (MobileNetV3), and a Transformer that looks at the picture as patches (ViT-B/16). These are the same five models as in the original paper.

| Model | In one line | Input size |
|---|---|---:|
| VGG16 | Classic design: many simple layers stacked. Big and slow, a useful old baseline. | 224 px |
| DenseNet121 | Each layer reuses features from earlier layers. Medium size. | 224 px |
| EfficientNet-B2 | Scales width, depth and image size together. Good accuracy for its size. | 256 px |
| MobileNetV3-Large | Very small and fast, designed for phones; this is the model for the on-phone test. | 224 px |
| ViT-B/16 | A Transformer: cuts the image into 16×16 patches and compares them. A different family from the four CNNs. | 224 px |

**How every run is trained (same recipe everywhere).**
- Photos are resized to the model's input size. During training only, they are randomly cropped, flipped, rotated by 90° and slightly colour-changed, so the model sees many variations.
- Each epoch the model sees a random sample that gives every class and every photo-group a fair share, so large groups of copies do not dominate.
- Optimiser AdamW with a different learning rate per model (the new output layer learns 10× faster), one warm-up epoch, then a gently falling rate; label smoothing 0.1 (stops the model being over-confident); at most 40 epochs, early stopping after 6 epochs without improvement on the validation set.
- The model with the best validation score is kept and then graded once on the test photos. Random seeds are fixed, so a run can be repeated.

**The eight kinds of training run ("protocols").** Each one answers a different question from the reviewer.

| Protocol | What it does | Why we ran it | Runs | Train h |
|---|---|---|---:|---:|
| `cv` | ko-pin only, group-disjoint 5-fold × 3 repeats (main) | The honest main result: no photo or copy of it is ever in both train and test. | 75 | 2.1 |
| `leaky` | submitted random image-level 70/15/15 split | Repeats the original paper's way of splitting, to measure how much it overstated performance. | 5 | 0.5 |
| `vendor` | Roboflow's own train/valid/test folders | A second view of the same leakage problem, using the dataset provider's own folders. | 5 | 0.5 |
| `xsrc` | train on each of 7 datasets alone, test on all others | Shows whether ko-pin is special, or whether training on any dataset works as well, and how far each dataset transfers. | 35 | 1.1 |
| `msrc` | pooled published-use datasets, leave-one-source-group-out | Independent-evidence check: pool the datasets used in published papers (ko-pin excluded) and see if they agree with ko-pin. | 25 | 2.8 |
| `cv_fixed5` | cv, repeat 0, fixed 5 epochs (ablation) | Checks that results do not depend on our choice to train until early stopping. | 25 | 0.4 |
| `cv_lr0.5` | cv r0_f0, learning rate ×0.5 | Checks that the learning-rate choice is not lucky (half the rate). | 5 | 0.3 |
| `cv_lr2` | cv r0_f0, learning rate ×2 | Same, for double the rate. | 5 | 0.3 |

## 2. Main result: careful cross-validation on ko-pin (`cv`)

**What it does.** ko-pin has about 43,000 image files, but they come from only about 4,100 distinct photos plus rotated, sheared and re-saved copies. We first found every group of near-identical files (using image fingerprints and a copy-detection model) and then built the 5 folds *by group*. A photo and its copies therefore always stay together: if one is in the test set, none of its copies is in training. We do this 3 times with different random cuts: **15 runs per model, 75 in total**. The validation and test sets contain one representative photo per group, so no group counts more than once.

**Why we used it.** This is the leakage-free replacement for the original experiment. The average over 15 runs, with its spread, tells us how good each model really is and how much the result wobbles from run to run.

**Classes.** All 7. Cleaning started from 3,150 groups and removed 355 groups traced to the same hospital as the Han data, 224 groups whose label was doubtful (pitting nails and splinter bleeding that had been filed as Beau's lines or black line, and a few alopecia photos) and 35 groups with conflicting labels, leaving 2,536 groups.

**How to read the tables.** *Macro-F1* is the main score. *Macro-F1 at ~100 px* repeats the test with the photos shrunk to about 100 pixels, the size of the Han hospital thumbnails. *Best epoch* is when early stopping found the best model.

| Model | Runs | Macro-F1 | Bal. acc. | Accuracy | Macro-F1 at ~100 px | Best epoch | Min/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGG16 | 15 | 0.883 ± 0.016 | 0.881 ± 0.019 | 0.913 ± 0.009 | 0.658 ± 0.030 | 16.9 | 2.3 |
| DenseNet121 | 15 | 0.895 ± 0.015 | 0.894 ± 0.016 | 0.922 ± 0.012 | 0.731 ± 0.034 | 12.7 | 1.3 |
| EfficientNet-B2 | 15 | 0.913 ± 0.019 | 0.910 ± 0.021 | 0.934 ± 0.010 | 0.709 ± 0.029 | 17.6 | 1.8 |
| MobileNetV3 | 15 | 0.901 ± 0.013 | 0.897 ± 0.014 | 0.925 ± 0.009 | 0.699 ± 0.029 | 20.0 | 1.3 |
| ViT-B/16 | 15 | 0.911 ± 0.019 | 0.909 ± 0.020 | 0.931 ± 0.014 | 0.863 ± 0.031 | 11.5 | 1.8 |

Average recall per class, over all cv runs (1.00 = every photo of that class found):

| Model | beau_s | black_line | clubbing | healthy_nail | onychomycosis | onycholysis | white_spot |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGG16 | 0.72 | 0.92 | 0.85 | 0.96 | 0.98 | 0.93 | 0.81 |
| DenseNet121 | 0.77 | 0.93 | 0.85 | 0.97 | 0.98 | 0.94 | 0.81 |
| EfficientNet-B2 | 0.78 | 0.94 | 0.90 | 0.98 | 0.99 | 0.95 | 0.83 |
| MobileNetV3 | 0.76 | 0.92 | 0.88 | 0.98 | 0.99 | 0.94 | 0.82 |
| ViT-B/16 | 0.79 | 0.94 | 0.89 | 0.96 | 0.97 | 0.95 | 0.86 |

## 3. How much did the original split overstate things? (`leaky`, `vendor` vs `cv`)

**What it does.** `leaky` copies the original paper: all 43,144 ko-pin files shuffled and cut 70/15/15 *by single image*, with no grouping. `vendor` uses the folders (train / valid / test) that the dataset provider supplied. In both, near-identical copies of test photos are in the training set: 99.4% of the `leaky` test images and 95% of the `vendor` test images have such a copy (`data/splits/leakage_audit.json`).

**Why we used it.** The reviewer suspected leakage. Training the same models on these splits and comparing with `cv` turns the suspicion into a measured number.

**Classes.** All 7, using every ko-pin file without the cleaning step.

**How to read it.** One run per model, so there is no spread. *Δ pts* is how many points higher the score is than the careful `cv` result. The gain is not a better model; the model is simply being graded on pictures it has already seen.

| Model | cv | leaky | Δ pts | vendor | Δ pts |
|---|---:|---:|---:|---:|---:|
| VGG16 | 0.883 | 0.997 | +11.5 | 0.996 | +11.3 |
| DenseNet121 | 0.895 | 0.997 | +10.2 | 0.993 | +9.8 |
| EfficientNet-B2 | 0.913 | 0.997 | +8.4 | 0.994 | +8.1 |
| MobileNetV3 | 0.901 | 0.998 | +9.7 | 0.994 | +9.3 |
| ViT-B/16 | 0.911 | 0.998 | +8.7 | 0.995 | +8.4 |

## 4. Do the cv models work on other datasets? (`ext_*`)

**What it does.** We take the models trained only on ko-pin (the 15 cv runs per model) and test them on photos from other sources: a hospital (Han), web sources, and several Kaggle datasets. Before testing we remove every photo that is a copy or near-copy of a ko-pin photo, so the models cannot have seen them. Each source is scored separately because their classes and image quality differ.

**Why we used it.** Good scores on ko-pin could mean the model learned ko-pin's particular cameras, backgrounds and style. Testing on outside data shows how well it would work on new photos in practice.

**Classes.** Only the classes each source has (column *k classes*). For example Han has 5 (no Beau's lines or clubbing), and farisfaikar and isratjahanmitu have 1 each.

**How to read it.** Macro-F1 is computed over the *k* classes in that source. **Be careful with the small sources:** after removing copies, some have very few photos (for example arthitaya has 53, mostly healthy nails, with a single photo for some classes), so their scores jump around. With only 1 class, the score mostly reflects how many photos were named correctly and cannot be compared with multi-class sources. The large, reliable test is Han A1 (17,697 images).

| Source | n | k classes | VGG16 | DenseNet121 | EfficientNet-B2 | MobileNetV3 | ViT-B/16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| arthitaya | 53 | 4 | 0.34 ± 0.05 | 0.56 ± 0.16 | 0.42 ± 0.09 | 0.46 ± 0.07 | 0.62 ± 0.11 |
| arthitaya2 | 2 | 2 | 0.52 ± 0.14 | 0.76 ± 0.27 | 0.53 ± 0.13 | 0.50 ± 0.00 | 0.87 ± 0.23 |
| farisfaikar | 118 | 1 | 0.82 ± 0.04 | 0.88 ± 0.02 | 0.86 ± 0.03 | 0.83 ± 0.04 | 0.91 ± 0.05 |
| han_A1 | 17,697 | 5 | 0.37 ± 0.02 | 0.42 ± 0.02 | 0.39 ± 0.03 | 0.39 ± 0.02 | 0.38 ± 0.04 |
| han_virtualE | 1,134 | 4 | 0.58 ± 0.02 | 0.63 ± 0.02 | 0.61 ± 0.02 | 0.59 ± 0.02 | 0.66 ± 0.03 |
| isratjahanmitu | 178 | 1 | 0.88 ± 0.01 | 0.89 ± 0.02 | 0.89 ± 0.01 | 0.91 ± 0.01 | 0.91 ± 0.03 |
| josephrasanjana | 367 | 2 | 0.84 ± 0.03 | 0.87 ± 0.02 | 0.88 ± 0.02 | 0.88 ± 0.02 | 0.83 ± 0.05 |
| kaggle17 | 10 | 2 | 0.59 ± 0.13 | 0.55 ± 0.12 | 0.47 ± 0.16 | 0.52 ± 0.14 | 0.77 ± 0.14 |
| nikhilgurav21 | 710 | 2 | 0.53 ± 0.06 | 0.55 ± 0.08 | 0.77 ± 0.05 | 0.67 ± 0.07 | 0.89 ± 0.04 |
| nuttida | 7 | 3 | 0.27 ± 0.17 | 0.51 ± 0.26 | 0.44 ± 0.23 | 0.65 ± 0.22 | 0.86 ± 0.15 |

## 5. Train on one dataset, test on all others (`xsrc`)

**What it does.** For each of 7 datasets we train a separate model on that dataset alone (5 models × 7 datasets = 35 runs). Each is graded on a held-out part of its own dataset and on every other dataset, again with any photo linked to the training data removed.

**Why we used it.** It is a backup plan for the reviewer's data-quality worry. If a model trained on dataset A does well on dataset B, the two agree. If only ko-pin-trained models do well, ko-pin is unusual. It also shows which datasets are good teachers.

**Classes.** Whatever each training dataset contains; they differ a lot:

| Trained on | Training images | Number of classes | Classes in its training data |
|---|---:|---:|---|
| kopin | 22,640 | 7 | beau_s, black_line, clubbing, healthy_nail, onychomycosis, onycholysis, white_spot |
| han | 19,833 | 5 | black_line, healthy_nail, onychomycosis, onycholysis, white_spot |
| han_virtualE | 1,189 | 4 | black_line, healthy_nail, onychomycosis, onycholysis |
| thai | 694 | 6 | beau_s, black_line, clubbing, healthy_nail, onycholysis, white_spot |
| nikhilgurav21 | 794 | 2 | clubbing, healthy_nail |
| kaggle17 | 135 | 4 | beau_s, clubbing, onycholysis, white_spot |
| josephrasanjana | 713 | 2 | healthy_nail, onychomycosis |

**How to read the matrix.** Rows = what the model was trained on; columns = what it was tested on; values = macro-F1 averaged over the 5 models. The first column (*own test*) is the dataset's own held-out part. On average it is 0.90 for the dataset's own test but 0.30 for other datasets. Two cautions: (1) a model cannot name a class it never saw, so datasets with only 2 classes (nikhilgurav21, josephrasanjana) must score badly on datasets that have other classes, while their own 2-class test is easy (0.98–0.99); (2) ko-pin is the only dataset with all 7 classes, which partly explains why it transfers best. A dash means the model was trained on that dataset.

| Trained on | own test | han_A1 | han_virtualE | arthitaya | nuttida | nikhilgurav21 | kaggle17 | josephrasanjana | kopin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kopin | 0.90 | 0.40 | 0.60 | 0.57 | 0.60 | 0.70 | 0.51 | 0.86 | – |
| han | 0.82 | – | 0.74 | 0.36 | 0.32 | 0.12 | 0.28 | 0.91 | 0.38 |
| han_virtualE | 0.82 | 0.57 | – | 0.29 | 0.25 | 0.24 | 0.11 | 0.93 | 0.34 |
| thai | 0.85 | 0.14 | 0.29 | – | – | 0.45 | 0.37 | 0.23 | 0.28 |
| nikhilgurav21 | 0.99 | 0.02 | 0.05 | 0.10 | 0.06 | – | 0.08 | 0.16 | 0.05 |
| kaggle17 | 0.90 | 0.03 | 0.07 | 0.25 | 0.32 | 0.15 | – | 0.00 | 0.14 |
| josephrasanjana | 0.98 | 0.27 | 0.34 | 0.08 | 0.00 | 0.23 | 0.00 | – | 0.12 |

## 6. Pool the published datasets, leave one group out (`msrc`)

**What it does.** We mix together all datasets that have been used in published papers (Han hospital, the Thai Kaggle sets, nikhilgurav21 and the 17-class Kaggle set). ko-pin is *not* in this pool because its source cannot be verified. Then in four runs per model we hold out one source group completely and test on it; in a fifth run nothing is held out (5 models × 5 runs = 25). Each model is also tested on clean ko-pin, which it never trained on.

**Why we used it.** It is a second, independent check on ko-pin. If models trained on other people's data also work on ko-pin, the ko-pin labels are consistent with the wider literature. It also shows how performance drops when a model meets a new source (domain shift).

**Classes.** All 7 are possible, but each pool is missing some, depending on what is held out:

| Held out | Pool size | Training images per class | Held-out test images per class | Test classes the pool lacks (< 20 images) |
|---|---:|---|---|---|
| han | 2,106 | beau_s 243, black_line 198, clubbing 893, healthy_nail 360, onycholysis 251, white_spot 161 | black_line 1,402, healthy_nail 12,152, onychomycosis 4,787, onycholysis 1,431, white_spot 156 | onychomycosis |
| thai | 26,419 | beau_s 2, black_line 2,151, clubbing 628, healthy_nail 14,646, onychomycosis 6,765, onycholysis 1,975, white_spot 252 | beau_s 83, black_line 78, clubbing 72, healthy_nail 54, onycholysis 85, white_spot 71 | beau_s |
| nikhilgurav21 | 26,464 | beau_s 246, black_line 2,334, clubbing 146, healthy_nail 14,349, onychomycosis 6,755, onycholysis 2,215, white_spot 419 | clubbing 462, healthy_nail 333 | – |
| kaggle17 | 27,062 | beau_s 137, black_line 2,355, clubbing 737, healthy_nail 14,678, onychomycosis 6,723, onycholysis 2,107, white_spot 325 | beau_s 34, clubbing 31, onycholysis 48, white_spot 24 | – |
| none (full pool) | 27,507 | beau_s 242, black_line 2,333, clubbing 882, healthy_nail 14,638, onychomycosis 6,800, onycholysis 2,203, white_spot 409 | – | – |

**How to read the scores.** Macro-F1 on the held-out group:

| Held-out group | VGG16 | DenseNet121 | EfficientNet-B2 | MobileNetV3 | ViT-B/16 |
|---|---:|---:|---:|---:|---:|
| han | 0.15 | 0.14 | 0.16 | 0.16 | 0.18 |
| thai | 0.35 | 0.37 | 0.39 | 0.40 | 0.43 |
| nikhilgurav21 | 0.17 | 0.26 | 0.22 | 0.27 | 0.54 |
| kaggle17 | 0.62 | 0.63 | 0.69 | 0.61 | 0.72 |

The low numbers are mostly a coverage problem, not a model problem: when Han is held out the pool has no onychomycosis at all (but the Han test has 4,787), and when Thai is held out it has just 2 Beau's-line images. So these runs measure how much the pooled data lacks, not how well models generalise; decide how to report them before the manuscript.


The same pooled models on clean ko-pin (1,176 images, never trained on):

| Held out of the pool | VGG16 | DenseNet121 | EfficientNet-B2 | MobileNetV3 | ViT-B/16 |
|---|---:|---:|---:|---:|---:|
| han | 0.22 | 0.29 | 0.22 | 0.26 | 0.33 |
| thai | 0.32 | 0.39 | 0.32 | 0.38 | 0.40 |
| nikhilgurav21 | 0.50 | 0.61 | 0.49 | 0.51 | 0.67 |
| kaggle17 | 0.33 | 0.43 | 0.42 | 0.40 | 0.51 |
| none (full pool) | 0.33 | 0.43 | 0.43 | 0.41 | 0.53 |

**Result.** These scores (0.22–0.67) are far below the 0.88–0.91 that models trained on ko-pin reach on ko-pin. So models trained on the other published datasets do not transfer well to ko-pin. From these runs alone we cannot tell whether the cause is a different photo style (domain shift), different labelling habits, or classes the pool lacks; it needs a closer look before it is written up.

## 7. Checks on the training recipe (ablations)

**What it does.** (a) *5-epoch ablation:* repeat the `cv` training of repeat 0 (5 folds) but stop every run after exactly 5 epochs, with no early stopping. (b) *Learning-rate sweep:* repeat the first `cv` fold with the learning rate halved and doubled.

**Why we used it.** The reviewer asked for the training choices to be justified. (a) shows whether our scores depend on how long we train. (b) shows that the learning rate we used is not a lucky pick. Both use only ko-pin and all 7 classes, like `cv`.

**5-epoch ablation** (macro-F1 on the test photos of repeat 0, 5 folds):

| Model | Early stopping (normal) | Fixed 5 epochs | Δ pts |
|---|---:|---:|---:|
| VGG16 | 0.885 ± 0.012 | 0.854 ± 0.021 | -3.1 |
| DenseNet121 | 0.900 ± 0.019 | 0.876 ± 0.008 | -2.5 |
| EfficientNet-B2 | 0.913 ± 0.015 | 0.868 ± 0.016 | -4.5 |
| MobileNetV3 | 0.897 ± 0.020 | 0.850 ± 0.017 | -4.7 |
| ViT-B/16 | 0.908 ± 0.023 | 0.933 ± 0.011 | +2.6 |

For the four CNNs, longer training helps. ViT-B/16 does as well or better with only 5 epochs, so for that model longer training did not help here.

**Learning-rate sweep** (first fold only, one run each; each cell is validation / test macro-F1; the middle column is the normal setting):

| Model | LR × 0.5 | LR × 1 | LR × 2 |
|---|---:|---:|---:|
| VGG16 | 0.891 / 0.914 | 0.897 / 0.898 | 0.893 / 0.870 |
| DenseNet121 | 0.922 / 0.900 | 0.929 / 0.917 | 0.923 / 0.909 |
| EfficientNet-B2 | 0.906 / 0.894 | 0.927 / 0.910 | 0.911 / 0.905 |
| MobileNetV3 | 0.901 / 0.924 | 0.914 / 0.929 | 0.928 / 0.955 |
| ViT-B/16 | 0.938 / 0.964 | 0.941 / 0.935 | 0.923 / 0.917 |

One run per cell, and the differences are about the same size as the normal fold-to-fold wobble (0.015–0.02), so this sweep does not show that any rate is clearly better.

## 8. Limits of these numbers

- Image labels were checked only for consistency with `data/manifests/index.csv`, not for medical correctness. No dermatologist was available.
- `leaky`, `vendor` and the learning-rate sweeps are single runs per model, so they have no error bars.
- Several outside datasets are small or have few classes (see sections 4–6), so their scores are unstable.
- `results/metrics/` still holds the partial analysis from 5 Nov. Run `bash run_all.sh analysis` before using any number in the manuscript (manuscript numbers must come from `results/macros.tex` / `results/tables/`).
