.PHONY: setup data index evalset eval eval-retrieval eval-generation report all api app test lint

UV ?= uv
RUN = $(UV) run --env-file .env

setup:            ## Install everything, including the embedding/reranking models' runtime
	$(UV) sync --all-extras
	@test -f .env || (cp .env.example .env && echo "Created .env: add ANTHROPIC_API_KEY and SEC_USER_AGENT")

data:             ## Download and parse the 10-K corpus from EDGAR (free, ~2 min)
	$(RUN) fcopilot download
	$(RUN) fcopilot parse

index:            ## Chunk with LLM context (~$10-20, asks first) and embed (CPU ~15 min)
	$(RUN) fcopilot chunk --config "+ filters" --dry-run
	@read -p "Proceed with LLM contextualization? [y/N] " ok && [ "$$ok" = y ]
	$(RUN) fcopilot chunk --config "+ filters"
	$(RUN) fcopilot index --config "+ filters"

evalset:          ## Generate the synthetic gold set with Claude (~$2), then review it
	$(RUN) fcopilot gen-gold
	@echo "Now review it: $(UV) run fcopilot review"

eval-retrieval:   ## Retrieval ablation ladder (LLM context step costs ~$10-20)
	$(RUN) fcopilot eval-retrieval --allow-llm-cost

eval-generation:  ## End-to-end answer quality with an LLM judge (~$10-15)
	$(RUN) fcopilot eval-generation

report:           ## Write results/REPORT.md and refresh the README results block
	$(RUN) fcopilot report

eval: eval-retrieval eval-generation report

all: data index evalset eval

api:              ## Serve the REST API on :8000
	$(RUN) uvicorn filings_copilot.api:app --port 8000

app:              ## Streamlit demo on :8501
	$(RUN) streamlit run app/streamlit_app.py

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
