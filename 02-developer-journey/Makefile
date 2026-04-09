# Makefile for Prompt Optimization Workshop Infrastructure
# Reads from .env file, overriding any shell environment variables

# Load .env file from project root and export variables (overriding existing env vars)
ifneq (,$(wildcard ../.env))
    include ../.env
    export $(shell sed 's/=.*//' ../.env | grep -v '^\#' | grep -v '^$$')
endif

# Disable AWS CLI pager (prevents requiring 'q' to exit)
export AWS_PAGER=

# Default values (used only if not set in .env)
AWS_REGION ?= us-east-1
INFRA_STACK_NAME ?= prompt-workshop-infra
COGNITO_STACK_NAME ?= prompt-workshop-cognito
LAMBDA_S3_BUCKET ?= $(shell AWS_PROFILE=$(AWS_PROFILE) aws sts get-caller-identity --query Account --output text)-$(AWS_REGION)-workshop-lambda

.PHONY: help deploy-all deploy-infra deploy-cognito delete-all delete-infra delete-cognito status check-env package-lambda create-bucket

help:
	@echo "Prompt Optimization Workshop - Infrastructure Deployment"
	@echo ""
	@echo "Usage:"
	@echo "  make deploy-all      Deploy all infrastructure (package, upload, infra, cognito)"
	@echo "  make deploy-infra    Deploy Lambda, S3, Knowledge Base"
	@echo "  make deploy-cognito  Deploy Cognito User Pool"
	@echo "  make package-lambda  Package Lambda code and layer"
	@echo "  make delete-all      Delete all infrastructure"
	@echo "  make delete-infra    Delete infrastructure stack"
	@echo "  make delete-cognito  Delete Cognito stack"
	@echo "  make status          Show stack status"
	@echo "  make check-env       Verify environment configuration"
	@echo ""
	@echo "Environment (from .env):"
	@echo "  AWS_PROFILE:         $(AWS_PROFILE)"
	@echo "  AWS_REGION:          $(AWS_REGION)"
	@echo "  INFRA_STACK_NAME:    $(INFRA_STACK_NAME)"
	@echo "  COGNITO_STACK_NAME:  $(COGNITO_STACK_NAME)"

check-env:
	@echo "Checking environment configuration..."
	@if [ -z "$(AWS_PROFILE)" ]; then \
		echo "ERROR: AWS_PROFILE not set in .env"; \
		exit 1; \
	fi
	@echo "AWS_PROFILE: $(AWS_PROFILE)"
	@echo "AWS_REGION: $(AWS_REGION)"
	@echo "Verifying AWS credentials..."
	@AWS_PROFILE=$(AWS_PROFILE) aws sts get-caller-identity --region $(AWS_REGION) || \
		(echo "ERROR: Invalid AWS credentials for profile $(AWS_PROFILE)"; exit 1)
	@echo "Environment OK"

create-bucket: check-env
	@echo "Creating S3 bucket for Lambda code: $(LAMBDA_S3_BUCKET)"
	@AWS_PROFILE=$(AWS_PROFILE) aws s3api head-bucket --bucket $(LAMBDA_S3_BUCKET) 2>/dev/null || \
		AWS_PROFILE=$(AWS_PROFILE) aws s3 mb s3://$(LAMBDA_S3_BUCKET) --region $(AWS_REGION)
	@echo "S3 bucket ready: $(LAMBDA_S3_BUCKET)"

package-lambda: check-env create-bucket
	@echo "Packaging Lambda code..."
	@mkdir -p .build
	@# Package Lambda function code (excluding the pre-packaged layer zip)
	@cd prerequisite/lambda/python && zip -r ../../../.build/lambda-code.zip . -x "*.pyc" -x "__pycache__/*" -x "ddgs-layer.zip"
	@echo "Lambda code packaged: .build/lambda-code.zip"
	@# Use pre-packaged DDGS layer from E2E sample
	@echo "Using pre-packaged DDGS layer..."
	@cp prerequisite/lambda/python/ddgs-layer.zip .build/ddgs-layer.zip
	@echo "DDGS layer ready: .build/ddgs-layer.zip"
	@# Upload to S3
	@echo "Uploading to S3..."
	@AWS_PROFILE=$(AWS_PROFILE) aws s3 cp .build/lambda-code.zip s3://$(LAMBDA_S3_BUCKET)/lambda-code.zip --region $(AWS_REGION)
	@AWS_PROFILE=$(AWS_PROFILE) aws s3 cp .build/ddgs-layer.zip s3://$(LAMBDA_S3_BUCKET)/ddgs-layer.zip --region $(AWS_REGION)
	@echo "Lambda artifacts uploaded to s3://$(LAMBDA_S3_BUCKET)/"

deploy-infra: check-env package-lambda
	@echo "Deploying infrastructure stack: $(INFRA_STACK_NAME)..."
	AWS_PROFILE=$(AWS_PROFILE) aws cloudformation deploy \
		--template-file prerequisite/infrastructure.yaml \
		--stack-name $(INFRA_STACK_NAME) \
		--region $(AWS_REGION) \
		--capabilities CAPABILITY_NAMED_IAM \
		--s3-bucket $(LAMBDA_S3_BUCKET) \
		--parameter-overrides \
			LambdaS3Bucket=$(LAMBDA_S3_BUCKET) \
			LambdaS3Key=lambda-code.zip \
			LayerS3Key=ddgs-layer.zip \
		--no-fail-on-empty-changeset
	@echo "Infrastructure deployed successfully"
	@echo ""
	@echo "Stack outputs:"
	@AWS_PROFILE=$(AWS_PROFILE) aws cloudformation describe-stacks \
		--stack-name $(INFRA_STACK_NAME) \
		--region $(AWS_REGION) \
		--query 'Stacks[0].Outputs' \
		--output table

deploy-cognito: check-env
	@echo "Deploying Cognito stack: $(COGNITO_STACK_NAME)..."
	AWS_PROFILE=$(AWS_PROFILE) aws cloudformation deploy \
		--template-file prerequisite/cognito.yaml \
		--stack-name $(COGNITO_STACK_NAME) \
		--region $(AWS_REGION) \
		--capabilities CAPABILITY_NAMED_IAM \
		--no-fail-on-empty-changeset
	@echo "Cognito deployed successfully"
	@echo ""
	@echo "Stack outputs:"
	@AWS_PROFILE=$(AWS_PROFILE) aws cloudformation describe-stacks \
		--stack-name $(COGNITO_STACK_NAME) \
		--region $(AWS_REGION) \
		--query 'Stacks[0].Outputs' \
		--output table

deploy-all: deploy-infra deploy-cognito
	@echo ""
	@echo "All infrastructure deployed successfully!"
	@echo ""
	@echo "Resources created:"
	@echo "  - Lambda function (web search)"
	@echo "  - S3 bucket with technical documentation"
	@echo "  - Bedrock Knowledge Base with S3 Vectors"
	@echo "  - Cognito User Pool for Gateway authentication"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run the workshop notebooks in order (01-07)"

delete-infra: check-env
	@echo "Deleting infrastructure stack: $(INFRA_STACK_NAME)..."
	AWS_PROFILE=$(AWS_PROFILE) aws cloudformation delete-stack \
		--stack-name $(INFRA_STACK_NAME) \
		--region $(AWS_REGION)
	@echo "Waiting for stack deletion..."
	AWS_PROFILE=$(AWS_PROFILE) aws cloudformation wait stack-delete-complete \
		--stack-name $(INFRA_STACK_NAME) \
		--region $(AWS_REGION)
	@echo "Infrastructure stack deleted"

delete-cognito: check-env
	@echo "Deleting Cognito stack: $(COGNITO_STACK_NAME)..."
	AWS_PROFILE=$(AWS_PROFILE) aws cloudformation delete-stack \
		--stack-name $(COGNITO_STACK_NAME) \
		--region $(AWS_REGION)
	@echo "Waiting for stack deletion..."
	AWS_PROFILE=$(AWS_PROFILE) aws cloudformation wait stack-delete-complete \
		--stack-name $(COGNITO_STACK_NAME) \
		--region $(AWS_REGION)
	@echo "Cognito stack deleted"

delete-all: delete-cognito delete-infra
	@echo "All infrastructure deleted"
	@echo "Note: S3 bucket $(LAMBDA_S3_BUCKET) was not deleted. Delete manually if needed."

status: check-env
	@echo "Infrastructure Stack Status:"
	@AWS_PROFILE=$(AWS_PROFILE) aws cloudformation describe-stacks \
		--stack-name $(INFRA_STACK_NAME) \
		--region $(AWS_REGION) \
		--query 'Stacks[0].{Name:StackName,Status:StackStatus,Created:CreationTime}' \
		--output table 2>/dev/null || echo "  Stack not found: $(INFRA_STACK_NAME)"
	@echo ""
	@echo "Cognito Stack Status:"
	@AWS_PROFILE=$(AWS_PROFILE) aws cloudformation describe-stacks \
		--stack-name $(COGNITO_STACK_NAME) \
		--region $(AWS_REGION) \
		--query 'Stacks[0].{Name:StackName,Status:StackStatus,Created:CreationTime}' \
		--output table 2>/dev/null || echo "  Stack not found: $(COGNITO_STACK_NAME)"

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf .build
	@echo "Done"
