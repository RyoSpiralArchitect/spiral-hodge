SELECT prompt_id, MIN(family) AS family,
       COUNT(*) AS valid_nodes,
       SUM(distance_supported) AS supported_nodes,
       AVG(recovery_cosine) AS recovery_cosine,
       AVG(cancellation_fraction) AS cancellation_fraction,
       AVG(effective_contributors) AS effective_contributors,
       AVG(weighted_relative_gap) AS weighted_relative_gap
FROM calibration_nodes
WHERE recovery_cosine IS NOT NULL
GROUP BY prompt_id
ORDER BY prompt_id;
