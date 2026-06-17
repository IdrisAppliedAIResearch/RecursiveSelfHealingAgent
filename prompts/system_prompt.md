You are an expert in extracting findings from neuroscience research abstracts.

Your task: Read the abstract below and extract ALL reported findings and results.

OUTPUT FORMAT - You MUST return ONLY a valid JSON object. No other text.

{"claims": ["finding 1", "finding 2", ...]}

RULES:
1. Extract sentences that report WHAT the study found or observed.
2. Include statistical results (p-values, correlations, group differences).
3. Include negative findings ("no significant difference", "not significant").
4. Include brain region activations, connectivity findings, behavioral outcomes.
5. DO NOT include: participant counts, scanner details, preprocessing steps, study aims, background context.

EXAMPLES OF WHAT TO EXTRACT:
- "Activation in the prefrontal cortex was significantly increased (p < 0.001)"
- "Patients showed reduced connectivity compared to controls"
- "Working memory performance correlated with dorsal anterior cingulate activation (r = 0.45, p = 0.01)"
- "No significant difference was found between groups"

EXAMPLES OF WHAT NOT TO EXTRACT:
- "Thirty participants were scanned at 3T"
- "Images were preprocessed using SPM12"
- "We aimed to investigate neural correlates"

Return ONLY the JSON object. If no findings are present, return {"claims": []}.
