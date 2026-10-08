-- BigQuery DDL & Seed: Model Pricing Reference Table
-- Decouples financial calculations from application code. Rates expressed in USD per 1M tokens.

CREATE TABLE IF NOT EXISTS `{{project_id}}.{{dataset_id}}.llm_pricing_reference`
(
    model_name STRING NOT NULL OPTIONS(description="Model family or exact identifier"),
    input_cost_per_million NUMERIC NOT NULL OPTIONS(description="USD cost per 1M prompt tokens"),
    output_cost_per_million NUMERIC NOT NULL OPTIONS(description="USD cost per 1M candidates tokens"),
    thinking_cost_per_million NUMERIC NOT NULL OPTIONS(description="USD cost per 1M thinking tokens"),
    cached_input_cost_per_million NUMERIC NOT NULL OPTIONS(description="USD cost per 1M cached tokens"),
    currency STRING NOT NULL DEFAULT 'USD',
    effective_start_date DATE NOT NULL,
    effective_end_date DATE
)
CLUSTER BY model_name;

-- Seed default pricing rates
INSERT INTO `{{project_id}}.{{dataset_id}}.llm_pricing_reference`
(model_name, input_cost_per_million, output_cost_per_million, thinking_cost_per_million, cached_input_cost_per_million, currency, effective_start_date, effective_end_date)
VALUES
('gemini-3.8-flash', 0.15, 0.60, 0.60, 0.0375, 'USD', '2026-01-01', NULL),
('gemini-3.5-flash-lite', 0.075, 0.30, 0.30, 0.01875, 'USD', '2026-01-01', NULL),
('gemini-3.1-pro-preview', 1.25, 5.00, 5.00, 0.3125, 'USD', '2026-01-01', NULL),
('gemini-embedding-001', 0.025, 0.00, 0.00, 0.00, 'USD', '2026-01-01', NULL);
