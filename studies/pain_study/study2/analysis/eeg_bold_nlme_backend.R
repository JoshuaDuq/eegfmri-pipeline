args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: eeg_bold_nlme_backend.R <data.tsv> <spec.tsv> <output.tsv>")
}

suppressPackageStartupMessages(library(nlme))

data_path <- args[[1]]
spec_path <- args[[2]]
output_path <- args[[3]]

data <- read.delim(
  data_path,
  sep = "\t",
  check.names = FALSE,
  stringsAsFactors = FALSE
)
spec <- read.delim(
  spec_path,
  sep = "\t",
  check.names = FALSE,
  stringsAsFactors = FALSE
)

if (nrow(spec) != 1) {
  stop("Expected exactly one model specification row.")
}

split_terms <- function(text) {
  if (is.na(text) || text == "") {
    return(character(0))
  }
  strsplit(text, "\\|")[[1]]
}

quote_name <- function(text) {
  paste0("`", text, "`")
}

write_result <- function(result_row) {
  out <- as.data.frame(result_row, check.names = FALSE, stringsAsFactors = FALSE)
  write.table(
    out,
    file = output_path,
    sep = "\t",
    row.names = FALSE,
    quote = TRUE
  )
}

predictor_column <- spec$predictor_column[[1]]
outcome_column <- spec$outcome_column[[1]]
outcome_variance_column <- spec$outcome_variance_column[[1]]
numeric_terms <- split_terms(spec$numeric_terms[[1]])
factor_terms <- split_terms(spec$factor_terms[[1]])
fit_method <- tolower(spec$fit_method[[1]])
max_iterations <- as.integer(spec$max_iterations[[1]])
em_iterations <- as.integer(spec$em_iterations[[1]])
singular_tolerance <- as.numeric(spec$singular_tolerance[[1]])

if (!(fit_method %in% c("reml"))) {
  stop("Unsupported fit_method; expected 'reml'.")
}

data$subject <- as.factor(data$subject)
data$run_num <- as.character(data$run_num)
data$trial_position <- as.numeric(data$trial_position)
data$trial_time <- as.numeric(data$trial_time)

fixed_terms <- c(
  quote_name(predictor_column),
  vapply(numeric_terms, quote_name, FUN.VALUE = character(1)),
  vapply(
    factor_terms,
    function(term) paste0("factor(", quote_name(term), ")"),
    FUN.VALUE = character(1)
  )
)

fixed_formula <- as.formula(
  paste(
    quote_name(outcome_column),
    "~",
    paste(fixed_terms, collapse = " + ")
  )
)
random_formula <- as.formula(
  paste("~", quote_name(predictor_column), "|", quote_name("subject"))
)
correlation_formula <- as.formula(
  paste(
    "~",
    quote_name("trial_time"),
    "|",
    quote_name("subject"),
    "/",
    quote_name("run_num")
  )
)

weights_struct <- NULL
if (!is.na(outcome_variance_column) && outcome_variance_column != "") {
  if (!(outcome_variance_column %in% colnames(data))) {
    stop(sprintf("Outcome variance column %s is missing from model data.", outcome_variance_column))
  }
  data[[outcome_variance_column]] <- as.numeric(data[[outcome_variance_column]])
  if (any(!is.finite(data[[outcome_variance_column]]) | data[[outcome_variance_column]] <= 0)) {
    stop("Outcome variance column must be finite and strictly positive.")
  }
  weights_struct <- varFixed(
    as.formula(paste("~", quote_name(outcome_variance_column)))
  )
}

warning_messages <- character(0)
fit <- withCallingHandlers(
  tryCatch(
    lme(
      fixed = fixed_formula,
      data = data,
      random = random_formula,
      correlation = corCAR1(form = correlation_formula),
      weights = weights_struct,
      method = toupper(fit_method),
      na.action = na.fail,
      control = lmeControl(
        maxIter = max_iterations,
        msMaxIter = max_iterations,
        niterEM = em_iterations,
        returnObject = TRUE
      )
    ),
    error = function(error) {
      structure(
        list(message = conditionMessage(error)),
        class = "fit_error"
      )
    }
  ),
  warning = function(warning) {
    warning_messages <<- c(warning_messages, conditionMessage(warning))
    invokeRestart("muffleWarning")
  }
)

if (inherits(fit, "fit_error")) {
  write_result(
    list(
      status = "model_not_interpretable",
      message = fit$message,
      converged = FALSE,
      singular = FALSE,
      beta = NA_real_,
      se = NA_real_,
      p_value = NA_real_,
      ci_low = NA_real_,
      ci_high = NA_real_,
      rho = NA_real_,
      loglik = NA_real_,
      aic = NA_real_,
      bic = NA_real_
    )
  )
  quit(save = "no", status = 0)
}

summary_table <- summary(fit)$tTable
if (!(predictor_column %in% rownames(summary_table))) {
  stop(sprintf("Predictor %s not found in nlme summary table.", predictor_column))
}

beta <- as.numeric(summary_table[predictor_column, "Value"])
se <- as.numeric(summary_table[predictor_column, "Std.Error"])
p_value <- as.numeric(summary_table[predictor_column, "p-value"])
df_value <- as.numeric(summary_table[predictor_column, "DF"])
critical_value <- suppressWarnings(qt(0.975, df = df_value))
if (!is.finite(critical_value)) {
  critical_value <- qnorm(0.975)
}
ci_low <- beta - critical_value * se
ci_high <- beta + critical_value * se

rho <- NA_real_
if (!is.null(fit$modelStruct$corStruct)) {
  rho_values <- suppressWarnings(
    as.numeric(coef(fit$modelStruct$corStruct, unconstrained = FALSE))
  )
  if (length(rho_values) > 0) {
    rho <- rho_values[[1]]
  }
}

covariance <- tryCatch(
  as.matrix(pdMatrix(fit$modelStruct$reStruct[[1]])),
  error = function(error) matrix(NA_real_, nrow = 0, ncol = 0)
)
if (length(covariance) == 0) {
  singular <- FALSE
} else {
  eigenvalues <- tryCatch(
    eigen((covariance + t(covariance)) / 2, symmetric = TRUE, only.values = TRUE)$values,
    error = function(error) NA_real_
  )
  singular <- any(!is.finite(eigenvalues)) || min(eigenvalues) <= singular_tolerance
}

convergence_patterns <- c(
  "iteration limit",
  "failed to converge",
  "false convergence",
  "singular convergence",
  "not positive definite"
)
converged <- TRUE
if (length(warning_messages) > 0) {
  for (pattern in convergence_patterns) {
    if (any(grepl(pattern, warning_messages, ignore.case = TRUE))) {
      converged <- FALSE
      break
    }
  }
}

status <- if (converged && !singular) "ok" else "model_not_interpretable"
message <- paste(unique(warning_messages), collapse = " || ")

write_result(
  list(
    status = status,
    message = message,
    converged = converged,
    singular = singular,
    beta = beta,
    se = se,
    p_value = p_value,
    ci_low = ci_low,
    ci_high = ci_high,
    rho = rho,
    loglik = as.numeric(logLik(fit)),
    aic = AIC(fit),
    bic = BIC(fit)
  )
)
