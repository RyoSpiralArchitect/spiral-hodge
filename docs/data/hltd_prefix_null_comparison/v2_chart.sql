WITH cells AS (
    SELECT prompt_id, prefix_length, component, CAST(alpha AS REAL) AS alpha,
           AVG(CAST(next_token_logprob_steered AS REAL)
               - CAST(next_token_logprob_base AS REAL)) AS delta_logprob,
           COUNT(*) AS seeds
    FROM v2_treatments
    GROUP BY prompt_id, prefix_length, component, CAST(alpha AS REAL)
), prompts AS (
    SELECT prompt_id, component, alpha, AVG(delta_logprob) AS delta_logprob,
           COUNT(*) AS prefixes, MIN(seeds) AS seeds
    FROM cells
    GROUP BY prompt_id, component, alpha
)
SELECT component, alpha,
       CASE WHEN alpha > 0 THEN '+' || CAST(alpha AS TEXT)
            ELSE CAST(alpha AS TEXT) END AS dose,
       AVG(delta_logprob) AS mean_delta_logprob,
       COUNT(*) AS prompts, MIN(prefixes) AS prefixes, MIN(seeds) AS seeds
FROM prompts
GROUP BY component, alpha
ORDER BY alpha, component;
