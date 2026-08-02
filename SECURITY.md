# Security policy

## Scope

This repository contains exploratory Python scripts, notebooks, source material, and generated Markdown artifacts. It includes code that can call external language-model services and code that loads machine-learning models.

## Credentials

The active GEN-24 scripts read `OPENAI_API_KEY` from a local `GEN-24/LLM_keys.key` file. This file is ignored by Git. Never commit API keys, tokens, private keys, `.env` files, or other credentials. If a credential is exposed, revoke/rotate it immediately with the relevant provider.

Do not add personal absolute filesystem paths or private source material to this public repository.

## Reporting a vulnerability

Please do not disclose an unpatched security issue in a public issue. Contact the Active Inference Institute through the contact method listed on its public website, https://www.activeinference.org, with a clear description, affected path, reproduction steps that do not include secrets, and an assessment of impact.

This repository does not promise a specific response time. Please allow maintainers reasonable time to investigate before public disclosure.
