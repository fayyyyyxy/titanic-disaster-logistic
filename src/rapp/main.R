# src/rapp/main.R
suppressPackageStartupMessages({
  library(readr)   # read_csv
  library(dplyr)
  library(tibble)
})

TRAIN <- "src/data/train.csv"
TEST  <- "src/data/test.csv"
OUT   <- "src/data/pred_test_r.csv"

features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")
target   <- "Survived"
num_cols <- c("Age","SibSp","Parch","Fare")

load_csv <- function(path, tag) {
  cat(sprintf("\n[LOAD-%s] %s\n", tag, path))
  if (!file.exists(path)) stop("File not found: ", path)
  df <- read_csv(path, show_col_types = FALSE)
  cat(sprintf("[LOAD-%s] shape=(%d, %d)\n", tag, nrow(df), ncol(df)))
  df
}

mode1 <- function(x) {
  ux <- x[!is.na(x)]
  if (!length(ux)) return(NA)
  names(sort(table(ux), decreasing = TRUE))[1]
}

prep_train <- function(df) {
  cat("\n[PREP] Using features: ", paste(features, collapse=", "), "\n", sep="")
  cat("[PREP] Strategy:\n")
  cat("  - Age → median impute\n")
  cat("  - Sex, Embarked → impute mode & one-hot\n")
  cat("  - Pclass, SibSp, Parch, Fare → numeric as-is\n")

  df <- df[, c(target, features)]

  num_meds <- vapply(df[num_cols], function(v) median(v, na.rm = TRUE), numeric(1))
  for (c in names(num_meds)) {
    if (anyNA(df[[c]])) {
      cat(sprintf("[IMPUTE-TRAIN] %s median = %g\n", c, num_meds[[c]]))
      df[[c]] <- ifelse(is.na(df[[c]]), num_meds[[c]], df[[c]])
    }
  }
   if (anyNA(df$Embarked)) {
    m <- mode1(df$Embarked); cat("[IMPUTE-TRAIN] Embarked mode = ", m, "\n", sep="")
    df$Embarked <- ifelse(is.na(df$Embarked), m, df$Embarked)
  }    
  # learn factor levels from train 
  sex_levels      <- sort(unique(df$Sex))
  embarked_levels <- sort(unique(df$Embarked))
  df$Sex      <- factor(df$Sex, levels = sex_levels)
  df$Embarked <- factor(df$Embarked, levels = embarked_levels)
  cat("[LEARNED] Sex levels:", paste(levels(df$Sex), collapse=", "), "\n")
  cat("[LEARNED] Embarked levels:", paste(levels(df$Embarked), collapse=", "), "\n")
  # one-hot via model.matrix
  form  <- ~ 0 + Pclass + Sex + Age + SibSp + Parch + Fare + Embarked
  mm_tr <- model.matrix(form, data = df, na.action = na.pass)

  # Build training DF
  train_glm_df <- as.data.frame(mm_tr)
  train_glm_df$Survived <- as.integer(df[[target]])
  cat(sprintf("[PREP] Train frame: %d rows, %d cols (incl. Survived)\n",
              nrow(train_glm_df), ncol(train_glm_df)))

  list(
    train_glm_df = train_glm_df,
    stats = list(
      num_meds = num_meds,
      sex_levels = sex_levels,
      embarked_levels = embarked_levels,
      form = form
    )
  )
}

prep_test <- function(df, stats) {
  df <- df[, features]

  for (c in names(stats$num_meds)) {
    if (anyNA(df[[c]])) {
      cat(sprintf("[IMPUTE-TEST] %s ← train median %g\n", c, stats$num_meds[[c]]))
      df[[c]] <- ifelse(is.na(df[[c]]), stats$num_meds[[c]], df[[c]])
    }
  }

  # apply train factor levels for Sex/Embarked
  df$Sex      <- factor(df$Sex, levels = stats$sex_levels)
  df$Embarked <- factor(df$Embarked, levels = stats$embarked_levels)

  mm_te <- model.matrix(stats$form, data = df, na.action = na.pass)
  cat(sprintf("[PREP-TEST] X_te: %d x %d (nrow(test)=%d)\n", nrow(mm_te), ncol(mm_te), nrow(df)))
  mm_te
}

main <- function() {
  cat("=== Titanic Prediction (R) ===\n")

  # 1) Load TRAIN
  train <- load_csv(TRAIN, "TRAIN")

  # 2) Prep TRAIN
  tr <- prep_train(train)

  # 3) Fit logistic regression (binomial GLM)
  cat("\n[TRAIN] Fitting logistic regression (binomial)…\n")
  # Build a data.frame for glm with y and X columns
  fit <- glm(Survived ~ ., data = tr$train_glm_df, family = binomial())
  cat("[TRAIN] Fit complete.\n")

  # Train accuracy
  p_tr <- predict(fit, newdata = tr$train_glm_df, type = "response")
  yhat_tr <- ifelse(p_tr >= 0.5, 1L, 0L)
  acc_tr <- mean(yhat_tr == tr$train_glm_df$Survived)
  cat(sprintf("[METRIC] TRAIN accuracy = %.4f\n", acc_tr))

  # Test → predict & save
  test <- load_csv(TEST, "TEST")
  X_te <- prep_test(test, tr$stats)
  cat("\n[TEST] Predicting on test.csv …\n")
  p_te <- predict(fit, newdata = as.data.frame(X_te), type = "response")
  yhat_te <- ifelse(p_te >= 0.5, 1L, 0L)

  # Should match exactly
  stopifnot(length(yhat_te) == nrow(test))

  out <- tibble(
    PassengerId = if ("PassengerId" %in% names(test)) test$PassengerId else seq_len(nrow(test)),
    Survived = as.integer(yhat_te)  
  )
  write_csv(out, OUT)
  cat(sprintf("[SAVE] Save predictions to %s\n", OUT))
  cat("\n[DONE]\n")
}

if (sys.nframe() == 0) main()
