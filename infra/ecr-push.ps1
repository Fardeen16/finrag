# Push finrag:local to ECR in us-east-1.
# Requires: Docker Desktop running, AWS CLI configured as the IAM admin user.
#
#   .\infra\ecr-push.ps1

$ErrorActionPreference = "Stop"
$Region = "us-east-1"
$Repo = "finrag"

$Identity = aws sts get-caller-identity --output json | ConvertFrom-Json
$Account = $Identity.Account
$Uri = "$Account.dkr.ecr.$Region.amazonaws.com/$Repo"

Write-Host "account $Account"
Write-Host "repo    $Uri"

$repoExists = $false
try {
    aws ecr describe-repositories --repository-names $Repo --region $Region | Out-Null
    if ($LASTEXITCODE -eq 0) { $repoExists = $true }
} catch {
    $repoExists = $false
}
if (-not $repoExists) {
    aws ecr create-repository --repository-name $Repo --region $Region --image-scanning-configuration scanOnPush=true | Out-Null
    Write-Host "created repository $Repo"
}

aws ecr get-login-password --region $Region |
    docker login --username AWS --password-stdin "$Account.dkr.ecr.$Region.amazonaws.com"

docker tag finrag:local "${Uri}:latest"
docker push "${Uri}:latest"
Write-Host "pushed ${Uri}:latest"
