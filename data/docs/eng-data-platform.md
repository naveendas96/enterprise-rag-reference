---
title: Data Platform Standards
roles: [engineering, admin]
---
# Data Platform Standards
The lakehouse uses a medallion architecture with bronze, silver and gold layers. Gold tables must have an owner, a data contract and freshness SLAs.

## Retention
Raw bronze data is retained for 400 days. Personally identifiable information must be tokenised before it reaches the silver layer.

## Deployment
Pipelines are deployed through CI with mandatory data quality tests; a failing expectation blocks promotion to production.
