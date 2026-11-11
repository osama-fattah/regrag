data "aws_caller_identity" "current" {}

locals {
  image_uri = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"

  base_env = {
    REGRAG_LLM        = var.llm_provider
    REGRAG_LLM_MODEL  = var.llm_model
    REGRAG_EMBEDDINGS = "hashing"
    REGRAG_INDEX_DIR  = "/app/.index"
    # audit records go to stdout -> CloudWatch Logs
    REGRAG_AUDIT_LOG   = "-"
    REGRAG_MAX_RETRIES = "4"
    AWS_LWA_PORT       = "8080"
    # lets /ask/stream send server-sent events
    AWS_LWA_INVOKE_MODE = "response_stream"
  }

  env = var.llm_provider == "gemini" ? merge(local.base_env, { GEMINI_API_KEY = var.gemini_api_key }) : local.base_env
}

# ---- container registry ------------------------------------------------------------------

resource "aws_ecr_repository" "app" {
  name                 = var.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 5 most recent images"

      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }

      action = { type = "expire" }
    }]
  })
}

# ---- IAM: least privilege ----------------------------------------------------------------

data "aws_iam_policy_document" "assume_lambda" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.name}-lambda"
  assume_role_policy = data.aws_iam_policy_document.assume_lambda.json
}

data "aws_iam_policy_document" "lambda" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["${aws_cloudwatch_log_group.lambda.arn}:*"]
  }

  # Bedrock access only when Bedrock is the provider; the Converse API is authorised by InvokeModel*.
  dynamic "statement" {
    for_each = var.llm_provider == "bedrock" ? [1] : []
    content {
      sid     = "InvokeBedrock"
      actions = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]

      resources = [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:*:${data.aws_caller_identity.current.account_id}:inference-profile/*",
      ]
    }
  }
}

resource "aws_iam_role_policy" "lambda" {
  name   = "${var.name}-lambda"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

# ---- function ----------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.name}"
  retention_in_days = var.log_retention_days
}

resource "aws_lambda_function" "app" {
  function_name = var.name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = local.image_uri
  architectures = ["x86_64"]
  memory_size   = var.memory_mb
  timeout       = 120

  environment {
    variables = local.env
  }

  logging_config {
    log_format = "Text"
    log_group  = aws_cloudwatch_log_group.lambda.name
  }

  depends_on = [aws_iam_role_policy.lambda]
}

resource "aws_lambda_function_url" "app" {
  function_name      = aws_lambda_function.app.function_name
  authorization_type = var.public_url ? "NONE" : "AWS_IAM"
  invoke_mode        = "RESPONSE_STREAM"

  cors {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST"]
    allow_headers = ["content-type", "x-user"]
  }
}

# Public (auth NONE) URL. If AWS also asks for lambda:InvokeFunction, scope it to calls made via the
# function URL rather than granting it to "*" outright.
resource "aws_lambda_permission" "public_url" {
  count                  = var.public_url ? 1 : 0
  statement_id           = "AllowPublicFunctionUrl"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.app.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

# ---- cost guardrail ----------------------------------------------------------------------

resource "aws_budgets_budget" "monthly" {
  count        = var.alert_email == "" ? 0 : 1
  name         = "${var.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.alert_email]
  }
}
