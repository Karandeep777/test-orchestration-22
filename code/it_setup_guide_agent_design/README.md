# IT Setup Guide Agent

## Overview
IT Setup Guide Agent is a professional, friendly, patient, encouraging, detail-oriented, supportive general agent designed for text interactions.

## Features


## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the agent:
```bash
python agent.py
```

## Configuration

The agent uses the following environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key (if using Anthropic)
- `GOOGLE_API_KEY`: Google API key (if using Google)

## Usage

```python
from agent import IT Setup Guide AgentAgent

agent = IT Setup Guide AgentAgent()
response = await agent.process_message("Hello!")
```

## Domain: general
## Personality: professional, friendly, patient, encouraging, detail-oriented, supportive
## Modality: text