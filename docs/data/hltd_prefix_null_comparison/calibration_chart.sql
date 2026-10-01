SELECT comparator,
       CASE comparator
           WHEN 'actual' THEN 'Actual coexact'
           WHEN 'v2_random' THEN 'Isotropic random'
           WHEN 'within_prompt_shuffle' THEN 'Within-prompt shuffle'
           WHEN 'within_prompt_position_shuffle' THEN 'Prompt + position-bin shuffle'
       END AS readout,
       AVG(CAST(control_cosine AS REAL)) AS mean_cosine,
       COUNT(*) AS draws,
       MIN(CAST(paired_nodes AS INTEGER)) AS paired_nodes,
       MIN(CAST(paired_prompts AS INTEGER)) AS paired_prompts
FROM calibration_draws
WHERE scope = 'all' AND matching = 'common' AND control_cosine <> ''
GROUP BY comparator
ORDER BY mean_cosine DESC;
