output "function_url" {
  description = "HTTPS endpoint: GET /health, POST /ask, POST /ask/stream"
  value       = aws_lambda_function_url.app.function_url
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "log_group" {
  value = aws_cloudwatch_log_group.lambda.name
}
