# HR Document Processor Agent

## Overview
HR Document Processor Agent is a professional, detail-oriented, reliable, compliant, transparent, efficient human_resources agent designed for text interactions.

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
from agent import HR Document Processor AgentAgent

agent = HR Document Processor AgentAgent()
response = await agent.process_message("Hello!")
```

## Domain: human_resources
## Personality: professional, detail-oriented, reliable, compliant, transparent, efficient
## Modality: text