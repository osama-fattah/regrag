variable "region" {
  description = "AWS region (Bedrock must offer your chosen model here)."
  type        = string
  default     = "eu-west-1"
}

variable "name" {
  description = "Name prefix for all resources."
  type        = string
  default     = "regrag"
}

variable "image_tag" {
  description = "Tag of the image in the ECR repository to deploy (pushed by scripts/deploy.sh)."
  type        = string
  default     = "latest"
}

variable "llm_provider" {
  description = "offline | bedrock | gemini | openai | azure"
  type        = string
  default     = "bedrock"

  validation {
    condition     = contains(["offline", "bedrock", "gemini", "openai", "azure"], var.llm_provider)
    error_message = "llm_provider must be one of offline, bedrock, gemini, openai, azure."
  }
}

variable "llm_model" {
  description = "Model id. For Bedrock, an inference profile or model id available in var.region."
  type        = string
  default     = "eu.anthropic.claude-3-5-haiku-20241022-v1:0"
}

variable "gemini_api_key" {
  description = "Only needed when llm_provider = gemini. Stored encrypted in the Lambda configuration."
  type        = string
  default     = ""
  sensitive   = true
}

variable "public_url" {
  description = "If true the function URL needs no auth (demo only). If false, callers must sign requests with AWS SigV4."
  type        = bool
  default     = false
}

variable "memory_mb" {
  description = "Lambda memory. 1024 MB keeps cold starts reasonable for the index load."
  type        = number
  default     = 1024
}

variable "monthly_budget_usd" {
  description = "Monthly account-level budget; alerts at 80% forecast."
  type        = number
  default     = 5
}

variable "alert_email" {
  description = "Email for budget alerts. Leave empty to skip the budget."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  type    = number
  default = 14
}
